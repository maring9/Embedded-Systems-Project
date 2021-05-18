# Importing needed libraries

# CV2 - Library used for computer vision operations
import cv2

# NUMPY - Library used for tensor operations
import numpy as np

# TENSORFLOW LITE RUNTIME - Light version of the Tensorflow Lite API
import tflite_runtime.interpreter as tflite


def preprocess_input(input_image):
    """Function to preprocess the input image as expected by the model

    Args:
        input_image (cv2.Image): Input image read from camera

    Returns:
        np.array: Tensor as expected by the model
    """

    # Convert the frame from default opencv BGR to RGB color space
    rgb_frame = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Resize the input to be in the format expected by the model
    resized_input = cv2.resize(rgb_frame, dsize=(224,224))

    # Add batch dimension to the input tensor, the model expects a 
    # tensor of rank 4
    input_tensor = np.expand_dims(resized_input, axis=0)

    # Normalize the tensor in the range [0, 1]
    input_tensor = input_tensor.astype(np.float32)
    input_tensor /= 255.

    return input_tensor


def load_model(model_path):
    """Load the model from storage

    Args:
        model_path (String): Absolute path where the model was saved

    Returns:
        tflite.Interpreter: Tensorflow Lite model
    """

    # Create Tensorflow Lite interpreter from the saved model
    interpreter = tflite.Interpreter(model_path)

    # Allocate tensors to run inference
    interpreter.allocate_tensors()

    return interpreter


def run_inference(interpreter, input_tensor):
    """Run inference on a model and input

    Args:
        interpreter (tflite.Interpreter): Tensorflow Lite model to run the inference
        input_tensor (np.array): Tensor of rank 4 to run the inference on

    Returns:
        np.array: Categorical vector containing the output from the model
    """

    # Input and output details from the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Give the input tensor to the first layer of the model
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Get output from the last layer of the model
    model_output = interpreter.get_tensor(output_details[0]['index'])

    return model_output


def classify_output(model_output):
    """Label the input image based on the model output

    Args:
        model_output (np.array): Categorical vector holding the output of the model

    Returns:
        String: Label
    """
    prediction = np.argmax(model_output)

    labels = {
        0:'Green Light',
        1:'Red Light'
    }

    return labels[prediction]


def main():
    # Get camera
    cap = cv2.VideoCapture(0)

    # Absolute path from where to load the tensorflow lite model
    model_path = '/home/pi/Desktop/Proiect/lite_model2.tflite'

    # Loading the model
    model = load_model(model_path)

    # Loop reads frame after frame from camera
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        input = preprocess_input(image)

        # Run inference
        output = run_inference(model, input)

        # Label the output
        label = classify_output(output)

        print(label)
        
        cv2.imshow('Image', image)
    
        # Press Q / ESC to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
