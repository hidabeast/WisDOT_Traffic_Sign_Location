import numpy as np

from dist_estimation import calculate_distance

def calculate_position(img_paths, img_data, HAOV, VAOV, VEHICLE_DELTA_Y, VEHICLE_DELTA_Z, img_sign_locs):
    distances = []
    for x in range(0, len(img_paths)): # calculates the distances for each picture
        distances.append(calculate_distance(img_paths[x], img_sign_locs[x], VEHICLE_DELTA_Z, VEHICLE_DELTA_Y, x))
    print("distances: ", [np.round(distance, 5) for distance in distances])

    coord_array = []
    for i in range(0, len(distances)): # find the 3D position of the sign in each image
        distance = distances[i] # feet
        sign_pxl_x = img_sign_locs[i][0][0] # i for each image, 0 for each sign, 0 for x coordinate
        sign_pxl_y = img_sign_locs[i][0][1] # i for each image, 0 for each sign, 1 for y coordinate
        midpoint_x = 8250 // 2
        midpoint_y = 2200 // 2
        horiz_deg_per_pxl = (HAOV / 2)/midpoint_x # calculates how many degrees to the right or left it is per pixel
        vert_deg_per_pxl = (VAOV / 2)/midpoint_y # calculates how many degrees up or down it is per pixel
        horiz_angle = abs(sign_pxl_x - midpoint_x)*horiz_deg_per_pxl # calculates the horizontal angle of the sign
        vert_angle = abs(sign_pxl_y - midpoint_y)*vert_deg_per_pxl # calculates the vertical angle of the sign

        theta = np.radians(horiz_angle) # theta is the fancy name for horizontal angle in spherical coordinates
        phi = np.radians(vert_angle) # phi is the fancy name for vertical angle in spherical coordinates

        # z is up and down, x is left and right, y is forward and backward
        cam_sign_x = distance * np.sin(theta)
        cam_sign_y = distance * np.cos(theta)
        cam_sign_z = distance * np.sin(phi)
        if (sign_pxl_x < midpoint_x): # if we are to the left of the midpoint
            cam_sign_x = -cam_sign_x
        if (sign_pxl_y < midpoint_y): # if we are under the midpoint
            cam_sign_z = -cam_sign_z # remember z here is altitude
        print("x cam", i + 1, ": ", np.round(cam_sign_x, 5), "y cam", i + 1, ": ", np.round(cam_sign_y, 5), "z cam", i + 1, ": ", np.round(cam_sign_z, 5))
        assert(np.abs(np.sqrt(cam_sign_x**2 + cam_sign_y**2 + cam_sign_z**2) - distance) < 1) # this makes sure
        # that the x, y, and z we calculated are relatively close to the distance(using pythagorean theorem in 3D)
        gps_sign_x = cam_sign_x # x should stay the same(gps config is right behind camera config)
        gps_sign_y = cam_sign_y + VEHICLE_DELTA_Y # y should increase(gps is further back)
        gps_sign_z = cam_sign_z + VEHICLE_DELTA_Z # z should increase(gps is below camera config)
        print("x gps", i + 1, ": ", np.round(gps_sign_x, 5), "y gps", i + 1, ": ", np.round(gps_sign_y, 5), "z gps", i + 1, ": ", np.round(gps_sign_z, 5))
        
        data = img_data[i]
        lat, lon, alt, yaw, pitch, roll = data[0], data[1], data[2], data[3], data[4], data[5]
        if yaw > 180:
            yaw = yaw - 360 # yaw should be positive or negative, not 0-360
        # we don't need to do this for pitch and roll because they are already like that
        yaw, pitch, roll = np.radians(yaw), np.radians(pitch), np.radians(roll)
        # note that the following rotational matrices are weird because we are using a weird coordinate system(x and y switched)
        Ry = np.array([[np.cos(yaw), np.sin(yaw), 0], # changed signs on the sines!
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]]) # rotation matrix for yaw
        Rp = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]]) # rotation matrix for pitch
        Rr = np.array([[np.cos(roll), 0, np.sin(roll)],
                    [0, 1, 0],
                    [-np.sin(roll), 0, np.cos(roll)]]) # rotation matrix for roll
        R = np.matmul(np.matmul(Ry, Rp), Rr) # final rotation matrix

        ### SANITY CHECK
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], # no changed signs
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]]) # rotation matrix for yaw (z-axis stays constant)
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]]) # rotation matrix for pitch (x-axis stays constant)
        Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                    [0, 1, 0],
                    [-np.sin(roll), 0, np.cos(roll)]]) # rotation matrix for roll (y-axis stays constant)
        R2 = np.matmul(np.matmul(Rz, Rx), Ry) # final rotation matrix
        assert(R.all() == R2.all())
        ### SANITY CHECK

        gps_sign_pos = np.array([[gps_sign_x], 
                                [gps_sign_y], 
                                [gps_sign_z]]) # 3x1 matrix representing sign x, y, z
        final_coords = np.matmul(R, gps_sign_pos) # apply rotation matrix to initial points
        final_coords = final_coords.reshape(1, 3)[0]
        final_coords = [np.round(coord, 5) for coord in final_coords]
        coord_array.append(final_coords)
        print(final_coords)
        # these coordinates are still not absolute. they are relative to the gps. they are more like displacements.
        # if we had coordinates in an x, y, z format for the location of the gps at each picture, then we could
        # add/subtract each x, y, z here to the gps point and get the point of the sign. 
        # our final coordinate system should be cartesian, where N to S is +y to -y, E to W is +x to -x,
        # and Up to Down is +z to -z.
    return coord_array

# GIVEN: Array of file locations to the images, camera HAOV(horizontal angle of view) and VAOV(vertical angle of view), 
# y and z displacement from camera to gps mount, latitude, longitude, altitude, yaw, pitch, and roll for each image
img_paths = ['/mnt/c/sign_1.png', '/mnt/c/sign_2.png',
            '/mnt/c/sign_3.png', '/mnt/c/sign_4.png',
            '/mnt/c/sign_5.png', '/mnt/c/sign_6.png',
            '/mnt/c/sign_7.png', '/mnt/c/sign_8.png'] # NEEDS TO BE AUTOMATED
img_data = [[43.912028, -88.029203, 894, 0, 0, 0],
            [43.911958, -88.029202, 894, 0, 0, 0],
            [43.911881, -88.029201, 894, 0, 0, 0],
            [43.911811, -88.029199, 894, 0, 0, 0],
            [43.911742, -88.029198, 893, 0, 0, 0],
            [43.911673, -88.029196, 892, 0, 0, 0],
            [43.911595, -88.029195, 892, 0, 0, 0],
            [43.911523, -88.029193, 891, 0, 0, 0]] # NEEDS TO BE CHANGED/AUTOMATED
HAOV = 84.21 # the combined horizontal angle of view of our cameras, in degrees
VAOV = 22.62 # the vertical angle of view of our cameras, in degrees
VEHICLE_DELTA_Y = 16 # in feet, the distance the gps is behind the cameras -- NEEDS TO BE CHANGED
VEHICLE_DELTA_Z = 9 # in feet, the distance the gps is below the cameras -- NEEDS TO BE CHANGED

# ASSUMED: 2D Array of x, y coordinates(pairs) for each sign detected in each image
# For now, we will only focus on the stop sign
img_sign_locs = [[(6296, 688)], [(5560, 728)],
                [(5240, 792)], [(5008, 800)],
                [(4832, 864)], [(4728, 880)],
                [(4672, 896)], [(4632, 888)]] # NEEDS TO BE AUTOMATED

relative_coordinates = calculate_position(img_paths, img_data, HAOV, VAOV, VEHICLE_DELTA_Y, VEHICLE_DELTA_Z, img_sign_locs)
print(relative_coordinates)

# MARK I
# distances: [30, 34, 35, 43, 40, 43, 45, 46]
# the problem I think is that the highest points are not even on the road, thus distorting the fit line

# MARK II
# changes: made only the first three points matter
# distances: [30.76, 33.69, 35.62, 44.26, 41.13, 44.73, 46.26, 49.65]

# MARK III
# changes: compensated for aspect ratio by adjusting y-values to image height(1327) and used all 6 points
# new known_y_coords: [2194, 1868, 1683, 1561, 1476, 1407]
# distances: [72.5, 112, 131.8, 178.7, 170.3, 183.6, 189.3, 201.7]
# positions relative to the camera:
# x 1 :  27.345 y 1 :  67.14   z 1 :  -5.355
# x 2 :  28.31  y 2 :  108.32  z 2 :  -7.468
# x 3 :  26.012 y 3 :  129.228 z 3 :  -7.28
# x 4 :  27.999 y 4 :  176.516 z 4 :  -9.616
# x 5 :  21.390 y 5 :  168.930 z 5 :  -7.209
# x 6 :  19.681 y 6 :  182.51  z 6 :  -7.245
# x 7 :  18.41  y 7 :  188.39  z 7 :  -6.92
# x 8 :  18.195 y 8 :  200.90  z 8 :  -7.672

# MARK IV
# changes: instead of calculating the distance from sign to gps mount, I subtracted the VEHICLE_DELTA_Y(how far
# away the camera is from the gps mount) from each known distance to get the distance from the sign to the camera.
# this is very important when finding the 3D coordinates of the sign
# TLDR: distances now contains distances from the sign to the camera instead of sign to gps mount
# distances: [60.568, 99.99, 120.04, 167.891, 159.175, 172.775, 178.64, 191.32]
# positions relative to the camera:
# x 1 :  22.845 y 1 :  56.094 z 1 :  -4.473
# x 2 :  25.283 y 2 :  96.735 z 2 :  -6.66
# x 3 :  23.68  y 3 :  117.67 z 3 :  -6.63
# x 4 :  26.30  y 4 :  165.81 z 4 :  -9.03
# x 5 :  19.995 y 5 :  157.91 z 5 :  -6.73
# x 6 :  18.524 y 6 :  171.77 z 6 :  -6.819
# x 7 :  17.38  y 7 :  177.7  z 7 :  -6.538
# x 8 :  17.257 y 8 :  190.540 z 8 :  -7.276
# still haven't compensated for pitch/roll/yaw

# MARK V
# changes: using the positions relative to the camera, I converted those to positions relative to the gps.
# from there, I calculate the rotation matrix using the yaw, pitch, and roll, and applied it to the coordinates.
# the results are now in terms of displacement(x, y, z) from the gps mount in feet.
# I also compensated for camera height when calculating distance to sign relative to camera in calculate_distance()
# and I did some rudimentary testing with different yaw/pitch/roll values. they rotations seem to be accurate.
# NOTE: I changed signs on two sines from the original yaw rotational matrix because it was wrong for some reason before that.
# NOTE: Resolved! I was multiplying the matrices wrong because it needs to be in a certain order and we
# are using a weird coordinate system!
# distances:  [57.1597, 96.55963, 116.65721, 164.67545, 155.93595, 169.60409, 175.51681, 188.26355]
# x gps 1 :  21.56030 y gps 1 :  68.937550 z gps 1 :  4.7778
# x gps 2 :  24.41708 y gps 2 :  109.42146 z gps 2 :  2.55885
# x gps 3 :  23.02045 y gps 3 :  130.36330 z gps 3 :  2.55551
# x gps 4 :  25.79788 y gps 4 :  178.64216 z gps 4 :  0.1389
# x gps 5 :  19.58862 y gps 5 :  170.70070 z gps 5 :  2.398
# x gps 6 :  18.18466 y gps 6 :  184.62641 z gps 6 :  2.30588
# x gps 7 :  17.07676 y gps 7 :  190.68410 z gps 7 :  2.57609
# x gps 8 :  16.98129 y gps 8 :  203.49613 z gps 8 :  1.83948
# positions relative to gps after rotation(with no rotation lol)
# [[21.5603, 68.93755, 4.7778], [24.41708, 109.42146, 2.55885], 
# [23.02045, 130.3633, 2.55551], [25.79788, 178.64216, 0.1389], 
# [19.58862, 170.7007, 2.398], [18.18466, 184.62641, 2.30588], 
# [17.07676, 190.6841, 2.57609], [16.98129, 203.49613, 1.83948]]
# issues: kind of hard to test this stuff. also I don't have the yaw/pitch/roll so I'm using 0, 0, 0. also low z values.
# note on testing: for the yaw at least for the eight testing photos, the van is going nearly northwards, 
# so yaw is probably close to 0.