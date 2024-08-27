import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.optimize import curve_fit
import torch
import math
import matplotlib.pyplot as plt


def calculate_distance(img_path, sign_locs, VEHICLE_DELTA_Z, VEHICLE_DELTA_Y, i): # finds distance to sign from camera
    # currently only works for arrays of size one, need to make it so it can calculate distance for multiple signs
    
    # FOR TESTING/TIME COMMENTING OUT SO THE MODEL DOESN'T MAKE A NEW DEPTH MAP EACH TIME
    # COMMENT BACK IN FOR PRODUCTION
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }

    # encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    # model = DepthAnythingV2(**model_configs[encoder])
    # model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    # model = model.to(DEVICE).eval()

    # raw_img = cv2.imread(img_path)
    # depth_map = model.infer_image(raw_img) # HxW raw depth map in numpy
    # depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    # depth_map_display = (depth_map_normalized*255).astype(np.uint8)
    # # cv2.imwrite("depth_map_" + str(i + 1) + ".png", depth_map_display)

    # Convert depth map to float32 and normalize (0-255 to 0-1 for processing)

    # COMMENTING IN
    depth_map_display = cv2.imread('depth_map_' + str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE)
    depth_map = depth_map_display.astype(np.float32) / 255.0

    # Known y-coordinates and corresponding distances
    known_y_coords_raw = np.array([1294, 1102, 993, 921, 871, 830]) # raw pixel coordinates(not adjusted)
    known_y_coords = np.array([int(round(y / 1327 * 2250)) for y in known_y_coords_raw]) # adjust to aspect ratio
    known_distances_gps = np.array([35, 48, 61, 74, 87, 100]) # distances relative to gps
    known_distances_cam = np.array([math.sqrt((dist - VEHICLE_DELTA_Y)**2 + VEHICLE_DELTA_Z**2)
                                        for dist in known_distances_gps]) # distances relative to camera
    # note that these distances are representing the distance from the camera to the spots on the road, so we need
    # to consider both the change in y and the change in z(because the camera is elevated)
    known_gray_values = np.array([depth_map[y, 4125] for y in known_y_coords])  # Grayscale values at known y-coordinates
    print("g_vs: ", known_gray_values)

    # Define an exponential model for fitting
    def exponential_model(gray, a, b, c):
        b = np.abs(b) # just in case b is negative(which it sometimes is)
        return a * np.power(b, gray) + c

    # Fit the model to the known data
    initial_guess = [217.1, 0.0728, 0]
    params, _ = curve_fit(exponential_model, known_gray_values, known_distances_cam, initial_guess, maxfev=10000)

    x_point = sign_locs[0][0] # first 0 hard-coded for one sign
    y_point = sign_locs[0][1] # first 0 hard-coded for one sign

    gray_value = depth_map[y_point, x_point]
    print("target g_v: ", gray_value)

    # Extrapolate the distance using the fitted model
    predicted_distance = exponential_model(gray_value, *params)

    # Save the plot of known points and fitted exponential curve
    plt.figure(figsize=(10, 6))
    plt.scatter(known_gray_values, known_distances_cam, color='red', label='Known Data')
    gray_fit = np.linspace(0, 1, 100)
    plt.plot(gray_fit, exponential_model(gray_fit, *params), color='blue', label='Fitted Exponential Curve')
    plt.xlabel('Grayscale Depth Value (0-1)')
    plt.ylabel('Distance (feet)')
    plt.title('Exponential Fit for Depth Estimation')
    plt.legend()
    plt.grid(True)
    plt.savefig('fit_plot_created.png')
    plt.close()

    return predicted_distance