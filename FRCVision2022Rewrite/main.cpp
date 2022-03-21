#include <iostream>
#include "librealsense2/rs.hpp"
#include "FindTarget.hpp"



int main() {
	std::cout << "Starting Program!" << std::endl;

	rs2::pipeline p;
	rs2::config cfg;

	cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 720, RS2_FORMAT_Y8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_Y8, 30);

	rs2::decimation_filter dec;
	dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);

	//rs2::disparity_transform depth2disparity;
	rs2::disparity_transform depth2disparity(false);

	rs2::spatial_filter spat;
	// 5 = fill all the zero pixels
	spat.set_option(RS2_OPTION_HOLES_FILL, 5);

	rs2::temporal_filter temp;

	auto profile = p.start(cfg);

	auto sensor = profile.get_device().first<rs2::depth_sensor>();

	sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
	sensor.set_option(RS2_OPTION_LASER_POWER, 150);


	rs2::align align_to_depth(RS2_STREAM_DEPTH);
	rs2::align align_to_ir(RS2_STREAM_INFRARED);
	rs2::align align_to_color(RS2_STREAM_COLOR);

	int frame_num = 0;

	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;

	const rs2_intrinsics cameraIntrinsics = p.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();


	while (true) {
		frame_num++;
		const auto before = clock::now();

		rs2::frameset frames = p.wait_for_frames();

		// align frames
		//frames = align_to_ir.process(frames);

		// get depth frame
		rs2::depth_frame depth = frames.get_depth_frame();
		depth = dec.process(depth);
		depth = depth2disparity.process(depth);
		depth = spat.process(depth);
		depth = temp.process(depth);

		// get ir frame
		rs2::frame img = frames.get_infrared_frame(1);

		// get color frame
		rs2::frame color = frames.get_color_frame();

		const int w = img.as<rs2::video_frame>().get_width();
		const int h = img.as<rs2::video_frame>().get_height();

		cv::Mat image = cv::Mat(cv::Size(w, h), CV_8UC1, (void*)img.get_data());

		cv::GaussianBlur(image, image, cv::Size(11, 5), 0, 0);

		threshold(image, image, 250, 255, cv::THRESH_BINARY);

		TargetFinder::TargetFinder targetFinder;

		TargetFinder::TargetData data = targetFinder.findTarget(image, depth, cameraIntrinsics);

		
		if (data.targetFound == false) {
			const sec duration = clock::now() - before;
			std::string sendableData = targetFinder.makeSendableData(0, 0, 0, 0, 0);
			targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
			std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
			std::cout << "Sending: " << sendableData << std::endl;
			continue;
		}


		const float distance = depth.get_distance(((data.centerTargetWidth / 2) + data.centerTarget.x) / 2, ((data.centerTargetHeight / 2) + data.centerTarget.y) / 2);
		const float distanceUnits = depth.as<rs2::depth_frame>().get_units();
		const double targetDepth = distance;

		const float targetPixel2D[2] = { (data.targetWidth / 2.0) + data.target.x, (data.targetHeight / 2.0) + data.target.y };
		float targetPoint3D[3] = { 0, 0, 0 };
		rs2_deproject_pixel_to_point(targetPoint3D, &cameraIntrinsics, targetPixel2D, targetDepth);

		cv::Vec3f normal = { 0,0,1 };
		cv::Vec3f targetVector = { targetPoint3D[0], 0, targetPoint3D[2] };
		const float dot = targetVector.dot(normal);
		const float targetVectorMag = sqrt(pow(targetVector[0], 2) + pow(targetVector[1], 2) + pow(targetVector[2], 2));
		float xTurnRad = acos(dot / (targetVectorMag));

		// If target is to the right, should be negative
		if ((data.target.x + (data.target.width / 2)) >= image.cols / 2) {
			xTurnRad *= -1;
		}

		const float distanceModifier = 1.31;
		float targetDepthAdjusted = targetDepth * (3.28084);
		const float targetHeightReal = 6.17;
		targetDepthAdjusted = sqrt(pow(targetDepthAdjusted, 2) - pow(targetHeightReal, 2));
		targetDepthAdjusted *= distanceModifier;
		targetDepthAdjusted = targetDepthAdjusted - 2.5;
		if (isnan(targetDepthAdjusted)) {
			const sec duration = clock::now() - before;
			std::string sendableData = targetFinder.makeSendableData(0, 0, 0, 0, 0);
			targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
			std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
			std::cout << "Sending: " << sendableData << std::endl;
			continue;
		}
		const float pitchA = 0.0078;
		const float pitchB = -0.2671;
		const float pitchC = -1.0;
		const float pitch = (pitchA * pow(targetDepthAdjusted, 2)) + (pitchB * targetDepthAdjusted) + pitchC;

		const float powerA = 8.33;
		const float powerB = -9.52;
		const float powerC = 3000;
		const float power = (powerA * pow(targetDepthAdjusted, 2)) + (powerB * targetDepthAdjusted) + powerC;

		const sec duration = clock::now() - before;

		std::cout << "Target Depth			: " << std::to_string(targetDepth) << " Meters" << "\n";
		std::cout << "Target Depth Adjusted	: " << std::to_string(targetDepthAdjusted) << " Ft" << "\n";
		std::cout << "X Turn Radians		: " << std::to_string(xTurnRad) << "\n";
		//cout << "Vector to Target		: " << to_string(targetPoint3D[0]) << " " << to_string(targetPoint3D[1]) << " " << to_string(targetPoint3D[2]) << endl;
		std::cout << "Y Turn Rotations		: " << std::to_string(pitch) << "\n";
		std::cout << "Turret Power			: " << std::to_string(power) << "\n";
		std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
		std::string sendableData = targetFinder.makeSendableData(data.targetFound, xTurnRad, pitch, power, targetDepthAdjusted);
		targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
		std::cout << "Sending: " << sendableData << std::endl;
#ifdef WINDOW
		cv::waitKey(1);
#endif
	}
	return 0;
}