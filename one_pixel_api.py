MODELAPI = "http://0.0.0.0:5000/model/predict"


def fgsm_attack_api(image, target_label=1, epsilon=0.1):
    """Performs FGSM attack using API calls instead of local model gradients."""
    
    image = image.astype(np.float32)  # Convert to float32
    perturbation = np.zeros_like(image)  # Initialize perturbation array
    delta = 1e-3  # Small perturbation for finite difference approximation

    # Compute numerical gradient for each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                perturbed_image = image.copy()
                perturbed_image[i, j, c] += delta  # Slightly increase pixel value

                original_pred = call_modelapi(image)["predictions"][0]["probability"]
                perturbed_pred = call_modelapi(perturbed_image)["predictions"][0]["probability"]

                # Compute approximate gradient (finite difference)
                gradient = (perturbed_pred - original_pred) / delta
                perturbation[i, j, c] = gradient

    # Apply adversarial perturbation in the direction of the gradient
    signed_grad = np.sign(perturbation)
    adversarial_image = image + epsilon * signed_grad  # Modify image
    adversarial_image = np.clip(adversarial_image, 0, 255).astype(np.uint8)  # Keep valid pixel values

    return adversarial_image


def call_modelapi(image_array):
    """Sends an image array to the API and returns the prediction."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Invalid input: Provide a valid NumPy image array.")

    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(np.uint8(image_array))

    # Resize to match model input size (if needed)
    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))

    # Convert image to bytes
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)

    # Send the image directly as a binary file
    files = {"image": ("image.png", buffer, "image/png")}
    response = requests.post(MODELAPI, files=files)

    return response.json()


def call_modelapi(image_path):
    # Ensure the input is a valid file path
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise ValueError("Invalid input: Provide a valid image file path.")

    # Open the image file in binary mode and send it
    with open(image_path, "rb") as img_file:
        files = {"image": (image_path, img_file, "image/png")}
        response = requests.post(MODELAPI, files=files)
    
    return response.json()  # Return the JSON response


def one_pixel_attackapi(image_path, preset_colors, max_iter=100):
    """Performs one-pixel attack to either minimize or maximize probability."""
    image = cv2.imread(image_path)  # Load image using OpenCV (BGR format)

    if image is None:
        raise ValueError("Error loading image. Check the file path.")

    height, width, _ = image.shape  # Get image dimensions

    # Step 1: Get the original probability before perturbation
    original_response = call_modelapi(image_path)
    original_prob = original_response["predictions"][0]["probability"]

    # Decide attack direction:
    # - If probability is near 1, we want to DECREASE it (Minimize).
    # - If probability is near 0, we want to INCREASE it (Maximize).
    attack_direction = -1 if original_prob > 0.5 else 1  # Flip objective

    def perturbation(params):
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]

        # Make a copy and modify one pixel
        img_copy = image.copy()
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        # Save perturbed image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_path = temp_img.name
            cv2.imwrite(temp_path, img_copy)

        # Call the model API with the modified image
        response = call_modelapi(temp_path)

        # Remove the temporary image file after sending
        os.remove(temp_path)

        # Attack direction determines whether we minimize or maximize
        return attack_direction * response["predictions"][0]["probability"]

    # Define bounds for pixel position and color choice
    bounds = [(0, width - 1), (0, height - 1), (0, len(preset_colors) - 1)]

    # Run the optimization
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

    # Extract the best perturbation found
    x, y, color_idx = int(result.x[0]), int(result.x[1]), int(result.x[2])
    r, g, b = preset_colors[color_idx % len(preset_colors)]

    return [x, y, r, g, b]