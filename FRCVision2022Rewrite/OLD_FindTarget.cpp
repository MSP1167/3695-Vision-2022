// FindTarget.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "boost/asio.hpp"
#include "boost/array.hpp"
#include "boost/bind.hpp"
#include <librealsense2/rs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
//#include "msmf.h"
#include <thread>
#include <ios>
#include <fstream>

#include <cstdlib>
#include <regex>

// Undefine to disable gui
#define WINDOW
// Line 569 = Depth Calc
// Vision Settings
#define VISION_AREA_OFFSET_X 70
#define VISION_AREA_OFFSET_Y 20
// Realsense FOV
#define REALSENSE_FOV_H 87
#define REALSENSE_FOV_V 58
// Jevois Cam ID
#define JEVOISID 0
// UDP
#define ROBOIPV4 "10.36.95.2"
#define ROBOPORT 7777
// Camera Calibration
#define CAMPIXELS 254
#define CAMDISTANCE 48
#define OBJECTWIDTH 11


using namespace std;
using namespace cv;
using boost::asio::ip::udp;
using boost::asio::ip::address;

class InputParser {
public:
	InputParser(int& argc, char** argv) {
		for (int i = 1; i < argc; ++i)
			this->tokens.push_back(std::string(argv[i]));
	}
	/// @author iain
	const std::string& getCmdOption(const std::string& option) const {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}
	/// @author iain
	bool cmdOptionExists(const std::string& option) const {
		return std::find(this->tokens.begin(), this->tokens.end(), option)
			!= this->tokens.end();
	}
private:
	std::vector <std::string> tokens;
};

Mat MatchTemplate(Mat& img, const Mat& templ, int match_method);
void findTarget(rs2::pipeline p, rs2::config cfg);
void gatherData(rs2::pipeline p, rs2::config cfg, int numFrames);
void findTargetSimple(rs2::pipeline p, rs2::config cfg, bool);
Rect getFullTargetRect(Rect mainRectangle, vector<Rect> rectangles, Point mainPoint, vector<Point> points);
void findTargetAdvanced(rs2::pipeline p, rs2::config cfg);
void findTargetJevois(int deviceId);
void listCameras();
void sendDataUDP(string in, string ip, int port);
string makeSendableData(int targetFound, double x, double y, double width, double groundDistance);
void convertPixelToPoint(float* point, float* pixel, float depth, rs2_intrinsics& cameraIntrinsics);
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth);

vector<array<double, 3>> readCSV();

int main2(int argc, char** argv)
{
	std::cout << "Starting!\n";

	//Mat img = imread(argv[1], 1); // Should be grabbed from cam image
	//Mat templ = imread(argv[2], 1); // Should be loaded from mem

	//Mat img = imread("C:\\Users\\Marc\\Documents\\Robotics\\test_Color.png", 1);
	//Mat templ = imread("C:\\Users\\Marc\\Documents\\Robotics\\test_Target.png", 1);

#ifdef _DEBUG
	cout << "Debug mode\n";
#endif // _DEBUG

#ifdef WINDOW
	namedWindow("Source Image", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);
#endif
	//namedWindow("Mask", WINDOW_AUTOSIZE);

	rs2::pipeline p;
	rs2::config cfg;

	cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 720, RS2_FORMAT_Y8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30);


	InputParser input(argc, argv);

	if (input.cmdOptionExists("-h")) {
		cout << "--demo			/ -d		| Run Demo" << endl;
		cout << "-g {f}						| Gather {f} number of frames" << endl;
		cout << "--simple					| Run Simple Test" << endl;
		cout << "--jevois {id}	/ -c {id}	| Use Jevois / USB Camera" << endl;
		return 0;
	}

	if (input.cmdOptionExists("-g")) {
		int numFrames = stoi(input.getCmdOption("-g"));
		cout << "Gathering for " << numFrames << " frames" << endl;
		gatherData(p, cfg, numFrames);
	}

	if (input.cmdOptionExists("-d") || input.cmdOptionExists("--demo")) {
		cout << "Running Demo" << endl;
		findTarget(p, cfg);
	}

	if (input.cmdOptionExists("--simple") || 1) {
		cout << "Using OPENCV VERSIOIN: " << CV_VERSION << endl;
		cout << "Running Simple Demo" << endl;
		findTargetSimple(p, cfg, false);
	}

	if (input.cmdOptionExists("--jevois") || input.cmdOptionExists("-c")) {
		cout << "Using OPENCV VERSIOIN: " << CV_VERSION << endl;
		cout << "Running Demo" << endl;
#ifdef _DEBUG
		//findTargetJevois(JEVOISID);
		return 0;
#endif
		int deviceId = stoi(input.getCmdOption("-c"));
		//findTargetJevois(deviceId);
	}

	//listCameras();
	//findTarget(p, cfg);

	//int match_method = 0;
	//Mat mask = MatchTemplate(img, templ, match_method);
	//imshow("Mask", mask);

	//cv::waitKey(0);

	return 0;
}

Mat MatchTemplate(Mat& img, const Mat& templ, int match_method)
{
	/// Source image to display
	Mat img_display; Mat result;
	if (img.channels() == 3)
		cvtColor(img, img, cv::COLOR_BGR2GRAY);
	img.copyTo(img_display);//for later show off

	/// Create the result matrix - shows template responces
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_cols, result_rows, CV_8UC1);

	/// Do the Matching and Normalize
	try {
		matchTemplate(img, templ, result, match_method);
	}
	catch (Exception) {
		//
	}
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;
	Point matchLoc;


	//in my variant we create general initially positive mask 
	Mat general_mask = Mat::ones(result.rows, result.cols, CV_8UC1);

	for (int k = 0; k < 5; ++k)// look for N=5 objects
	{
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, general_mask);
		//just to visually observe centering I stay this part of code:
		result.at<float>(minLoc) = 1.0;//
		result.at<float>(maxLoc) = 0.0;//

		// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. 
		 //For all the other methods, the higher the better
		if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
			matchLoc = minLoc;
		else
			matchLoc = maxLoc;
		//koeffitient to control neiboring:
		//k_overlapping=1.- two neiboring selections can overlap half-body of     template
		//k_overlapping=2.- no overlapping,only border touching possible
		//k_overlapping>2.- distancing
		//0.< k_overlapping <1.-  selections can overlap more then half 
		float k_overlapping = 1.7f;//little overlapping is good for my task

		//create template size for masking objects, which have been found,
		//to be excluded in the next loop run
		int template_w = ceil(k_overlapping * templ.cols);
		int template_h = ceil(k_overlapping * templ.rows);
		int x = matchLoc.x - template_w / 2;
		int y = matchLoc.y - template_h / 2;

		//shrink template-mask size to avoid boundary violation
		if (y < 0) y = 0;
		if (x < 0) x = 0;
		//will template come beyond the mask?:if yes-cut off margin; 
		if (template_w + x > general_mask.cols)
			template_w = general_mask.cols - x;
		if (template_h + y > general_mask.rows)
			template_h = general_mask.rows - y;

		//set the negative mask to prevent repeating
		Mat template_mask = Mat::zeros(template_h, template_w, CV_8UC1);
		template_mask.copyTo(general_mask(cv::Rect(x, y, template_w, template_h)));

		/// Show me what you got on main image and on result (
		rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
		//small correction here-size of "result" is smaller
		rectangle(result, Point(matchLoc.x - templ.cols / 2, matchLoc.y - templ.rows / 2), Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2), Scalar::all(0), 2, 8, 0);
	}//for k= 0--5 
#ifdef WINDOW
	imshow("Source Image", img_display);
#endif
	//imshow("Result", result);

	return result;
}

void findTarget(rs2::pipeline p, rs2::config cfg) {
	auto profile = p.start(cfg);

	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
	while (true) {
		// wait for frames
		rs2::frameset frames = p.wait_for_frames();

		// get depth frame
		rs2::depth_frame depth = frames.get_depth_frame();

		// get ir frame
		rs2::frame img = frames.get_infrared_frame(1);

		// convert to opencv
		const int w = img.as<rs2::video_frame>().get_width();
		const int h = img.as<rs2::video_frame>().get_height();

		Mat image = Mat(Size(w, h), CV_8UC1, (void*)img.get_data());
#ifdef WINDOW
		imshow("Source Image", image);
		//cv::waitKey(0);
#endif
		Mat binary;

		threshold(image, binary, 200, 255, THRESH_BINARY);
#ifdef WINDOW
		imshow("binary", binary);
		//cv::waitKey(0);
#endif

		std::vector<Point> locations;

		findNonZero(binary, locations);

		int x = 0;
		int y = 0;

		for (int i = 0; i < locations.size(); i++) {
			Point pnt = locations[i];

			x += pnt.x;
			y += pnt.y;
		}

		int avg_x = x / locations.size();
		int avg_y = y / locations.size();

		Mat color;

		cvtColor(image, color, COLOR_GRAY2RGB);

		Scalar circle_color = (255, 255, 255);

		circle(color, Point(avg_x, avg_y), 10, circle_color);
#ifdef WINDOW
		imshow("Source Image", color);
		//cv::waitKey(0);
#endif
		float distanceToTarget = depth.get_distance(avg_x, avg_y);

		putText(color, to_string(distanceToTarget), Point(avg_x, avg_y), FONT_HERSHEY_PLAIN, 1, circle_color, 1, 8);
#ifdef WINDOW
		imshow("Source Image", color);
		//cv::waitKey(0);
#endif

	}
}

void gatherData(rs2::pipeline p, rs2::config cfg, int numFrames) {
	cfg.enable_record_to_file("data.bag");
	auto profile = p.start(cfg);
#ifdef WINDOW
	cv::destroyWindow("Source Image");
#endif
	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
	int framecount = 0;
	while (true) {
		framecount++;
		// wait for frames
		rs2::frameset frames = p.wait_for_frames();

		// get depth frame
		rs2::depth_frame depth = frames.get_depth_frame();

		// get ir frame
		rs2::frame img = frames.get_infrared_frame(1);

		if (framecount >= numFrames) {
			cout << "Ended on frame " << framecount << endl;
			break;
		}

	}
}

void findTargetSimple(rs2::pipeline p, rs2::config cfg, bool doOne) {

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
	//rs2::align align_to_color(RS2_STREAM_INFRARED);

	//vector<array<double, 3>> CSVDATA = readCSV();
	// Focal Length of Camera
	// FocLen = (Pixels * Distance) / Width | Width should be inch.
	//double focalLength = (CAMPIXELS * CAMDISTANCE) / OBJECTWIDTH;
	int frame_num = 0;
#ifdef WINDOW
	int key = pollKey();
#endif

	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;

	while (true) {
		// Start Clock
		const auto before = clock::now();
		// swap laser if needed
#ifdef WINDOW
		//cout << key << endl;
		if (key == 49) {
			// 0 is off
			sensor.set_option(RS2_OPTION_LASER_POWER, 0);
		}
		// use this setting for normal use
		if (key == 50) {
			// 360 is max
			sensor.set_option(RS2_OPTION_LASER_POWER, 150);
		}
		if (key == 51) {
			// 360 is max
			sensor.set_option(RS2_OPTION_LASER_POWER, 360);
		}
#endif

		// wait for frames
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

		// convert to opencv
		const int w = img.as<rs2::video_frame>().get_width();
		const int h = img.as<rs2::video_frame>().get_height();

		Mat image = Mat(Size(w, h), CV_8UC1, (void*)img.get_data());
#ifdef WINDOW
		imshow("Source Image", image);
#endif
		GaussianBlur(image, image, Size(11, 5), 0, 0);

		Mat binary;

		threshold(image, binary, 250, 255, THRESH_BINARY);


		//imshow("binary", binary);
		//waitKey(0);
		bitwise_not(binary, binary);
		// Setup SimpleBlobDetector parameters.
		SimpleBlobDetector::Params params;

		// Minimum Distance Between Blobs
		params.minDistBetweenBlobs = 25.0f;

		// Change thresholds
		params.minThreshold = 200;
		params.maxThreshold = 255;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = 10;
		params.maxArea = 1000;

		// Filter by Circularity
		params.filterByCircularity = false;
		params.minCircularity = 0.1;

		// Filter by Convexity
		params.filterByConvexity = false;
		params.minConvexity = 0.87;

		// Filter by Inertia
		params.filterByInertia = false;
		params.minInertiaRatio = 0.01;

		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

		std::vector<KeyPoint> keypoints;
		detector->detect(binary, keypoints);

		Mat DetectedTargetsImage;
		drawKeypoints(image, keypoints, DetectedTargetsImage, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);



		// Convert Keypoints to Points
		vector<Point> points;
		for (int i = 0; i < keypoints.size(); i++)
		{
			points.push_back(Point(keypoints[i].pt.x, keypoints[i].pt.y));
		}

		// Before doing rects. we need to filter by distance
		vector<Point> betterPoints;
		for (Point betterPoint : points) {
			if (depth.get_distance(betterPoint.x / 2, betterPoint.y / 2) < 6) {
				betterPoints.push_back(betterPoint);
			}
		}


		vector<Rect> rectangles;
		for (Point point : betterPoints) {
			//line(image, Point(0, keypoints[i].pt.y), Point(image.cols, keypoints[i].pt.y), (0, 0, 255));
			Point2f corner1 = Point(point.x - VISION_AREA_OFFSET_X, point.y + VISION_AREA_OFFSET_Y);
			Point2f corner2 = Point(point.x + VISION_AREA_OFFSET_X, point.y - VISION_AREA_OFFSET_Y);
			Rect rect1 = Rect(corner1, corner2);
			rectangles.push_back(rect1);
			rectangle(DetectedTargetsImage, rect1, (0, 0, 255));
		}
		//imshow("Keypoints", DetectedTargetsImage);

		vector<int> pointsInside(rectangles.size(), 0);
		for (int i = 0; i < rectangles.size(); i++) {
			for (int j = 0; j < betterPoints.size(); j++) {
				if (rectangles[i].contains(betterPoints[j])) {
					pointsInside[i]++;
				}
			}
#ifdef _DEBUG
			cout << "Points in rectangle " << i << ": " << pointsInside[i] << endl;
#endif
		}
		// Check if Vector is Empty
		if (pointsInside.empty()) {
			continue;
		}
		// Find first Max Value
		int maxvalue = pointsInside[0];
		int position = 0;
		for (unsigned int i = 0; i < pointsInside.size(); i++) {
			if (pointsInside[i] > maxvalue) {
				maxvalue = pointsInside[i];
				position = i;
			}
		}

		const int highestPointValue = maxvalue;
		const int highestPointValueLocation = position;
		Rect target = getFullTargetRect(rectangles[highestPointValueLocation], rectangles, betterPoints[highestPointValueLocation], betterPoints);

		Mat Output;
		cvtColor(image, Output, COLOR_GRAY2RGB);
		rectangle(Output, target, (255, 0, 255));

		// show images for debug
#ifdef _DEBUG
		// re-invert binary image
		bitwise_not(binary, binary);
#ifdef WINDOW
		imshow("binary", binary);
		imshow("Keypoints", DetectedTargetsImage);
#endif
		bitwise_not(binary, binary);
#endif
#ifdef WINDOW
		key = waitKey(10);
#endif

		// If we do not see 3 points, continue as there is no point in finding target
		if (betterPoints.size() < 3) {
			continue;
		}

		// Find Target #'s
		// 1. Get width of target and known target size
		// 2. Get scale factor
		// 3. Get Depth to Target (Through Lookup / Math)
		// 4. Get X Distance using Scale & Distance to Target Center
		// 5. Do tan(X/Depth) to get X Turn Radians
		// 6. Use Depth to get Y Radians
		// 7. Use Depth to get Power
		// 8. Send data

		const Rect centerTarget = rectangles[highestPointValueLocation];
		const double targetWidth = target.width;
		const double targetHeight = target.height;
		const double centerTargetWidth = centerTarget.width;
		const double centerTargetHeight = centerTarget.height;
		//int realTargetWidth = 48; // Target is 48 in.
		//int scaleFactor = targetWidth / realTargetWidth;
		// Distance to Target is linear, and therefor we can just

		// csv should be in y, power, distance
		//vector<array<double, 3>> data = CSVDATA;
		// Mathematical way of determining depth
		//double targetDepth = scaleFactor * focalLength * targetWidth;

		const int w_aligned = depth.as<rs2::video_frame>().get_width();
		const int h_aligned = depth.as<rs2::video_frame>().get_height();

		Mat alignedDepth(Size(w, h), CV_8UC1, (void*)depth.get_data());

		// fancy way of determining depth
		// Divide by 2 on x and y? (((centerTargetWidth / 2) + centerTarget.x) / 2, ((centerTargetHeight / 2) + centerTarget.y) / 2);
		const float distance = depth.get_distance(((centerTargetWidth / 2) + centerTarget.x) / 2, ((centerTargetHeight / 2) + centerTarget.y) / 2);
		circle(Output, Point((targetWidth / 2) + target.x, (targetHeight / 2) + target.y), 5, (0, 255, 0), 2);
		//float distance = 3000;
		circle(Output, Point((targetWidth / 2) + target.x, (targetHeight / 2) + target.y), 5, (255, 0, 155), 2);
		const float distanceUnits = depth.as<rs2::depth_frame>().get_units();
		const double targetDepth = distance;
		//double xTurnRad = atan2((target.width / 2.0), targetDepth);
		// Horizontal Angle = ((x-W/2)/(W/2))(HFOV/2)

		const rs2_intrinsics cameraIntrinsics = p.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

		// Find Target X Turn
		// 1. Deproject depth to 3D coords
		// 2. Make vectors out of points
		// 3. TurnX = arccos[(vec1 * vec2)/(|vec1||vec2|)]
		//double targetX = ((targetWidth / 2) + target.x);
		//double depthWidthOver2 = (depth.get_width() / 2.0);
		//double xTurnRad = (((targetX - depthWidthOver2) / depthWidthOver2) * (REALSENSE_FOV_H / 2));

		const float targetPixel2D[2] = { (targetWidth / 2.0) + target.x, (targetHeight / 2.0) + target.y };
		float targetPoint3D[3] = { 0, 0, 0 };
		rs2_deproject_pixel_to_point(targetPoint3D, &cameraIntrinsics, targetPixel2D, targetDepth);

		//rs2::pointcloud pc;
		//rs2::points pointsOnCloud = pc.calculate(depth);
		//const rs2::vertex* vertices = pointsOnCloud.get_vertices();
		//for (int i = 0; i < pointsOnCloud.size(); i++)
		//{
		//	cout << "Vertex " << i << ": " << "[x: " << vertices[i].x << ", y: " << vertices[i].y << ", z: " << vertices[i].z << "]" << endl;
		//}
		//convertPixelToPoint(targetPixel2D, targetPoint3D, targetDepth, cameraIntrinsics);

		Vec3f normal = { 0,0,1 };
		Vec3f targetVector = { targetPoint3D[0], 0, targetPoint3D[2] };
		const float dot = targetVector.dot(normal);
		const float targetVectorMag = sqrt(pow(targetVector[0], 2) + pow(targetVector[1], 2) + pow(targetVector[2], 2));
		float xTurnRad = acos(dot / (targetVectorMag));

		// If target is to the right, should be negative
		if ((target.x + (target.width / 2)) >= image.cols / 2) {
			xTurnRad *= -1;
		}

		// OLD CODE FOR FINDING DISTANCE BASED OFF CSV
		// find equation of line from point before and point after
		/*
		array<double, 3> lowPoint;
		array<double, 3> highPoint;
		for (int i = 0; i < (data.size() - 1); i++)
		{
			if (data[i][2] <= targetDepth) {
				lowPoint = data[i];
			}
			else {
				highPoint = data[i + 1];
			}

		}
		// [0] is y, [1] is power
		array<double, 2> slopes;
		// change in y over change in x
		slopes[0] = (lowPoint[0] - highPoint[0]) / (lowPoint[2] - highPoint[2]);
		slopes[1] = (lowPoint[1] - highPoint[1]) / (lowPoint[2] - highPoint[2]);

		double yTurnRad = slopes[0] * targetDepth;
		double shootingPower = slopes[1] * targetDepth;
		*/

		// Find Distance / Y Pos based off math equation
		const float distanceModifier = 1.31;
		float targetDepthAdjusted = targetDepth * (3.28084);
		const float targetHeightReal = 6.17;
		targetDepthAdjusted = sqrt(pow(targetDepthAdjusted, 2) - pow(targetHeightReal, 2));
		targetDepthAdjusted *= distanceModifier;
		targetDepthAdjusted = targetDepthAdjusted - 2.5;
		if (isnan(targetDepthAdjusted)) {
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

		cout << "Target Depth			: " << to_string(targetDepth) << " Meters" << endl;
		cout << "Target Depth Adjusted		: " << to_string(targetDepthAdjusted) << " Ft" << endl;
		cout << "X Turn Radians			: " << to_string(xTurnRad) << endl;
		//cout << "Vector to Target		: " << to_string(targetPoint3D[0]) << " " << to_string(targetPoint3D[1]) << " " << to_string(targetPoint3D[2]) << endl;
		cout << "Y Turn Rotations		: " << to_string(pitch) << endl;
		cout << "Turret Power			: " << to_string(power) << endl;
		frame_num++;
		cout << "Frame Number	: " << to_string(frame_num) << " Took " << duration.count() << "s" << endl;
		string sendableData = makeSendableData(1, xTurnRad, pitch, power, targetDepthAdjusted);
		sendDataUDP(sendableData, ROBOIPV4, ROBOPORT);
		cout << "Sending: " << sendableData << endl;

		// show images for debug
#ifdef _DEBUG
		// draw extra info
		String depthInfo = "Depth: " + to_string(targetDepth);
		Size depthInfoSize = getTextSize(depthInfo, FONT_HERSHEY_DUPLEX, 1, 2, 0);
		putText(Output, depthInfo, Point(0, depthInfoSize.height), FONT_HERSHEY_DUPLEX, 1, CV_RGB(188, 185, 0), 2);
		String xInfo = "X Turn: " + to_string(xTurnRad);
		Size xInfoSize = getTextSize(xInfo, FONT_HERSHEY_DUPLEX, 1, 2, 0);
		putText(Output, xInfo, Point(0, depthInfoSize.height + xInfoSize.height), FONT_HERSHEY_DUPLEX, 1, CV_RGB(188, 185, 0), 2);
#endif

#ifdef WINDOW
		imshow("Output", Output);
		waitKey(10);
#endif

		if (doOne) {
			return;
		}
	}
}

Rect getFullTargetRect(Rect mainRectangle, vector<Rect> rectangles, Point mainPoint, vector<Point> points) {
	// get left bound
	Rect bigRectangle = mainRectangle;
	for (int j = points.size() - 1; j >= 0; j--)
	{
		if (bigRectangle.contains(points[j])) {
			bigRectangle = bigRectangle | rectangles[j];
		}
	}
	// get right bound
	for (int j = 0; j < points.size(); j++)
	{
		if (bigRectangle.contains(points[j])) {
			bigRectangle = bigRectangle | rectangles[j];;
		}
	}
	return bigRectangle;
}

void findTargetAdvanced(rs2::pipeline p, rs2::config cfg) {
	findTargetSimple(p, cfg, true);
	auto profile = p.get_active_profile();

	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
	sensor.set_option(RS2_OPTION_LASER_POWER, 130);


}

/*void findTargetJevois(int deviceId) {
	Mat frame;
	VideoCapture cap;

	int deviceID = deviceId;
	int apiID = cv::CAP_DSHOW;

	cap.open(deviceID);

	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return;
	}

	vector<array<double, 3>> CSVDATA = readCSV();
	// Focal Length of Camera
	// FocLen = (Pixels * Distance) / Width | Width should be inch.
	double focalLength = (CAMPIXELS * CAMDISTANCE) / OBJECTWIDTH;

	// Grab / Action Loop
	for (;;) {
		// wait for new frame
		cap.read(frame);
		// skip if empty
		if (frame.empty()) {
			cerr << "ERROR: EMPTY FRAME READ!" << endl;
			break;
		}

		// Find Target
		Mat image = frame;
#ifdef WINDOW
		imshow("Source Image", image);
		//waitKey(10);
#endif

		Mat binary;

		threshold(image, binary, 250, 255, THRESH_BINARY);

		//imshow("binary", binary);
		//waitKey(0);
		bitwise_not(binary, binary);
		// Setup SimpleBlobDetector parameters.
		SimpleBlobDetector::Params params;

		// Change thresholds
		params.minThreshold = 200;
		params.maxThreshold = 255;

		// Filter by Area.
		params.filterByArea = true;
		params.minArea = 10;

		// Filter by Circularity
		params.filterByCircularity = false;
		params.minCircularity = 0.1;

		// Filter by Convexity
		params.filterByConvexity = false;
		params.minConvexity = 0.87;

		// Filter by Inertia
		params.filterByInertia = false;
		params.minInertiaRatio = 0.01;

		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

		std::vector<KeyPoint> keypoints;
		detector->detect(binary, keypoints);

		Mat DetectedTargetsImage;
		drawKeypoints(image, keypoints, DetectedTargetsImage, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		// If we do not see 3 points, continue as there is no point in finding target
		if (keypoints.size() < 3) {
			continue;
		}
		// Convert Keypoints to Points
		vector<Point> points;
		for (int i = 0; i < keypoints.size(); i++)
		{
			points.push_back(Point(keypoints[i].pt.x, keypoints[i].pt.y));
		}

		vector<Rect> rectangles;
		for (int i = 0; i < keypoints.size(); i++) {
			//line(image, Point(0, keypoints[i].pt.y), Point(image.cols, keypoints[i].pt.y), (0, 0, 255));
			Point2f corner1 = Point(keypoints[i].pt.x - VISION_AREA_OFFSET_X, keypoints[i].pt.y + VISION_AREA_OFFSET_Y);
			Point2f corner2 = Point(keypoints[i].pt.x + VISION_AREA_OFFSET_X, keypoints[i].pt.y - VISION_AREA_OFFSET_Y);
			Rect rect1 = Rect(corner1, corner2);
			rectangles.push_back(rect1);
			rectangle(DetectedTargetsImage, rect1, (0, 0, 255));
		}

		vector<int> pointsInside(rectangles.size(), 0);
		for (int i = 0; i < rectangles.size(); i++) {
			for (int j = 0; j < points.size(); j++)
			{
				if (rectangles[i].contains(points[j])) {
					pointsInside[i]++;
				}
			}
			cout << "Points in rectangle " << i << ": " << pointsInside[i] << endl;
		}

		// Find first Max Value
		int maxvalue = pointsInside[0];
		int position = 0;
		for (unsigned int i = 0; i < pointsInside.size(); i++) {
			if (pointsInside[i] > maxvalue) {
				maxvalue = pointsInside[i];
				position = i;
			}
		}

		int highestPointValue = maxvalue;
		int highestPointValueLocation = position;
		Rect target = getFullTargetRect(rectangles[highestPointValueLocation], rectangles, points[highestPointValueLocation], points);

		Mat Output;
		cvtColor(image, Output, COLOR_GRAY2RGB);
		rectangle(Output, target, (255, 0, 255));

		// Find Target #'s
		// 1. Get width of target and known target size
		// 2. Get scale factor
		// 3. Get Depth to Target (Through Lookup / Math)
		// 4. Get X Distance using Scale & Distance to Target Center
		// 5. Do tan(X/Depth) to get X Turn Radians
		// 6. Use Depth to get Y Radians
		// 7. Use Depth to get Power
		// 8. Send data

		double targetWidth = target.width;
		int realTargetWidth = 48; // Target is 48 in.
		int scaleFactor = targetWidth / realTargetWidth;
		// Distance to Target is linear, and therefor we can just

		// csv should be in y, power, distance
		vector<array<double, 3>> data = CSVDATA;
		double targetDepth = scaleFactor * focalLength * targetWidth;

		double xTurnRad = atan2((target.width / 2.0) , targetDepth);

		// find equation of line from point before and point after
		array<double, 3> lowPoint;
		array<double, 3> highPoint;
		for (int i = 0; i < (data.size() - 1); i++)
		{
			if (data[i][2] <= targetDepth) {
				lowPoint = data[i];
			}
			else {
				highPoint = data[i + 1];
			}

		}
		// [0] is y, [1] is power
		array<double, 2> slopes;
		// change in y over change in x
		slopes[0] = (lowPoint[0] - highPoint[0]) / (lowPoint[2] - highPoint[2]);
		slopes[1] = (lowPoint[1] - highPoint[1]) / (lowPoint[2] - highPoint[2]);

		double yTurnRad			= slopes[0] * targetDepth;
		double shootingPower	= slopes[1] * targetDepth;

		string sendableData = makeSendableData(1, xTurnRad, yTurnRad, shootingPower);
		sendDataUDP(sendableData, ROBOIPV4, ROBOPORT);


		// show images for debug
#ifdef _DEBUG
		// re-invert binary image
		bitwise_not(binary, binary);
#ifdef WINDOW
		imshow("binary", binary);
		imshow("Keypoints", DetectedTargetsImage);
#endif
		// draw extra info
		String depthInfo = "Depth: " + to_string(targetDepth);
		Size depthInfoSize = getTextSize(depthInfo, FONT_HERSHEY_DUPLEX, 1, 2, 0);
		putText(Output, depthInfo, Point(0,0), FONT_HERSHEY_DUPLEX, 1, CV_RGB(188,185,0), 2);
		String xInfo = "X Turn: " + to_string(xTurnRad);
		putText(Output, xInfo, Point(0, depthInfoSize.height), FONT_HERSHEY_DUPLEX, 1, CV_RGB(188, 185, 0), 2);
#endif
#ifdef WINDOW
		imshow("Output", Output);
		//waitKey(10);
#endif
	}
	return;
}*/
/*
void listCameras() {
	msmf::DeviceEnumerator cameras;

	map camerasMap = cameras.getVideoDevicesMap();
	for (auto const& [key, val] : camerasMap) {
		std::cout << key << ": " << val.id << " : " << val.deviceName << endl;
	}

	cout << endl;
	return;
}
*/
void sendDataUDP(string in, string ip, int port)
{
	boost::asio::io_service io_service;
	udp::socket socket(io_service);
	udp::endpoint remote_endpoint = udp::endpoint(address::from_string(ip), port);
	socket.open(udp::v4());

	boost::system::error_code err;
	auto sent = socket.send_to(boost::asio::buffer(in), remote_endpoint, 0, err);
	socket.close();
	std::cout << "Sent Payload --- " << sent << "\n";
}

string makeSendableData(int targetFound, double x, double y, double power, double groundDistance) {
	string data = to_string(targetFound) + "," + to_string(x) + "," + to_string(y) + "," + to_string(power) + "," + to_string(groundDistance);
	return data;
}

vector<array<double, 3>> readCSV() {
	vector<array<double, 3>> finalData;
	//finalData.push_back({0, 0, 0});
	// set dummy data incase read fails
	//const char* test = "Test, Data, 9";
	//ofstream("testData.csv", ios::out).write(test, sizeof test);

	// y, power, distance
	ifstream dataFile("targetData.csv", ios::in);
	if (!dataFile.is_open()) {
		cerr << "COULD NOT OPEN DATA FILE! ERROR: " << strerror(errno) << endl;
		return finalData;
	}
	string line;
	vector<string> csvObjects;
	regex re(",");
	while (getline(dataFile, line)) {
		std::copy(                          // We want to copy something
			std::sregex_token_iterator      // The iterator begin, the sregex_token_iterator. Give back first token
			(
				line.begin(),               // Evaluate the input string from the beginning
				line.end(),                 // to the end
				re,                         // Add match a comma
				-1                          // But give me back not the comma but everything else 
			),
			std::sregex_token_iterator(),   // iterator end for sregex_token_iterator, last token + 1
			std::back_inserter(csvObjects)  // Append everything to the target container
		);
	}

	// Make 2d vector
	if (csvObjects.size() % 3 != 0) {
		cerr << "ERROR PARSING CSV!" << endl;
		return finalData;
	}

	int csvLine = 0;
	for (int i = 0; i < csvObjects.size() / 3; i++)
	{
		finalData.push_back(array<double, 3> { 0, 0, 0 });
	}
	for (int i = 0; i < csvObjects.size(); i++)
	{

		if (((i % 3) == 0) && (i != 0)) {
			csvLine++;
		}
		finalData[csvLine][i % 3] = stod(csvObjects[i]);
	}
	return finalData;
}

static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}
