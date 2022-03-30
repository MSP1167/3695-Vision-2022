#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include "librealsense2/rs.hpp"
#include "FindTarget.hpp"
#include <opencv2/videoio.hpp>
#include <iostream>
#include <boost/asio/ip/udp.hpp>
#include <boost/asio/ip/network_v4.hpp>
#include <boost/asio/io_service.hpp>

using boost::asio::ip::udp;
using boost::asio::ip::address;

static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth);
cv::Rect getFullTargetRect(cv::Rect mainRectangle, std::vector<cv::Rect> rectangles, cv::Point mainPoint, std::vector<cv::Point> points);

/// <summary>
/// Take a Black/White Image and Find a Segmented Target, Uses depth to target to find if target is valid
/// </summary>
/// <param name="img">Base Image to Find Target From</param>
/// <param name="depth">Depth Image to Get Image Data From</param>
/// <param name="cameraIntrinsics">Camera Intrinsics</param>
/// <returns>2D Array of Points with Center of Target and Center of Center Target</returns>
TargetFinder::TargetData TargetFinder::TargetFinder::findTarget(cv::Mat img, rs2::depth_frame depth, rs2_intrinsics cameraIntrinsics)
{
	TargetData targetData;

	targetData.targetFound = false;

	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;

#ifdef WINDOW
	cv::imshow("Source Image", img);
#endif

	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

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

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img, keypoints);

	drawKeypoints(img, keypoints, img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

#ifdef WINDOW
	cv::imshow("Keypoints", img);
#endif

	// Convert Keypoints to Points
	std::vector<cv::Point> points;
	for (cv::KeyPoint keypoint : keypoints)
	{
		points.push_back(cv::Point(keypoint.pt.x, keypoint.pt.y));
	}


	// Before doing rects. we need to filter by distance
	std::vector<cv::Point> betterPoints;
	for (cv::Point betterPoint : points) {
		// If point is farther than 6 meters, discard
		if (depth.get_distance(betterPoint.x / 2, betterPoint.y / 2) < 6) {
			betterPoints.push_back(betterPoint);
		}
	}

	// If we do not see 3 points, return as there is no reliable target to find
	if (betterPoints.size() < 3) {
		return targetData;
	}


	std::vector<cv::Rect> rectangles;
	for (cv::Point point : betterPoints) {
		//line(image, Point(0, keypoints[i].pt.y), Point(image.cols, keypoints[i].pt.y), (0, 0, 255));
		cv::Point2f corner1 = cv::Point(point.x - VISION_AREA_OFFSET_X, point.y + VISION_AREA_OFFSET_Y);
		cv::Point2f corner2 = cv::Point(point.x + VISION_AREA_OFFSET_X, point.y - VISION_AREA_OFFSET_Y);
		cv::Rect rect1 = cv::Rect(corner1, corner2);
		rectangles.push_back(rect1);
		rectangle(img, rect1, (0, 0, 255));
	}
#ifdef WINDOW
	cv::imshow("Detected Targets", img);
#endif

	std::vector<int> pointsInside(rectangles.size(), 0);
	for (int i = 0; i < rectangles.size(); i++) {
		for (cv::Point point : betterPoints) {
			if (rectangles[i].contains(point)) {
				pointsInside[i]++;
			}
		}
#ifdef _DEBUG
		std::cout << "Points in rectangle " << i << ": " << pointsInside[i] << "\n";
#endif
	}

	// Check if Vector is Empty
	if (pointsInside.empty()) {
		return targetData;
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
	
	targetData.targetFound = true;
	targetData.target = getFullTargetRect(rectangles[highestPointValueLocation], rectangles, betterPoints[highestPointValueLocation], betterPoints);
	targetData.centerTarget = rectangles[highestPointValueLocation];
	targetData.targetWidth = targetData.target.width;
	targetData.targetHeight = targetData.target.height;
	targetData.centerTargetWidth = targetData.centerTarget.width;
	targetData.centerTargetHeight = targetData.centerTarget.height;

	return targetData;
}

TargetFinder::TargetData TargetFinder::TargetFinder::findTargetNoDepth(cv::Mat img)
{
	TargetData targetData;

	targetData.targetFound = false;

	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;

	cv::bitwise_not(img, img);

#ifdef WINDOW
	cv::imshow("Source Image", img);
#endif

	

	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

	// Minimum Distance Between Blobs
	params.minDistBetweenBlobs = 25.0f;

	// Change thresholds
	params.minThreshold = 200;
	params.maxThreshold = 255;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 10000;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img, keypoints);

	drawKeypoints(img, keypoints, img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

#ifdef WINDOW
	cv::imshow("Keypoints", img);
#endif

	// Convert Keypoints to Points
	std::vector<cv::Point> points;
	for (cv::KeyPoint keypoint : keypoints)
	{
		points.push_back(cv::Point(keypoint.pt.x, keypoint.pt.y));
	}

	// If we do not see 3 points, return as there is no reliable target to find
	if (points.size() < 3) {
		return targetData;
	}


	std::vector<cv::Rect> rectangles;
	for (cv::Point point : points) {
		//line(image, Point(0, keypoints[i].pt.y), Point(image.cols, keypoints[i].pt.y), (0, 0, 255));
		cv::Point2f corner1 = cv::Point(point.x - VISION_AREA_OFFSET_X, point.y + VISION_AREA_OFFSET_Y);
		cv::Point2f corner2 = cv::Point(point.x + VISION_AREA_OFFSET_X, point.y - VISION_AREA_OFFSET_Y);
		cv::Rect rect1 = cv::Rect(corner1, corner2);
		rectangles.push_back(rect1);
		rectangle(img, rect1, (0, 0, 255));
	}
#ifdef WINDOW
	cv::imshow("Detected Targets", img);
#endif

	std::vector<int> pointsInside(rectangles.size(), 0);
	for (int i = 0; i < rectangles.size(); i++) {
		for (cv::Point point : points) {
			if (rectangles[i].contains(point)) {
				pointsInside[i]++;
			}
		}
#ifdef _DEBUG
		std::cout << "Points in rectangle " << i << ": " << pointsInside[i] << "\n";
#endif
	}

	// Check if Vector is Empty
	if (pointsInside.empty()) {
		return targetData;
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

	targetData.targetFound = true;
	targetData.target = getFullTargetRect(rectangles[highestPointValueLocation], rectangles, points[highestPointValueLocation], points);
	targetData.centerTarget = rectangles[highestPointValueLocation];
	targetData.targetWidth = targetData.target.width;
	targetData.targetHeight = targetData.target.height;
	targetData.centerTargetWidth = targetData.centerTarget.width;
	targetData.centerTargetHeight = targetData.centerTarget.height;

	return targetData;
}

cv::Rect getFullTargetRect(cv::Rect mainRectangle, std::vector<cv::Rect> rectangles, cv::Point mainPoint, std::vector<cv::Point> points) {
	// get left bound
	cv::Rect bigRectangle = mainRectangle;
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

void TargetFinder::TargetFinder::sendDataUDP(std::string in, std::string ip, int port)
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

std::string TargetFinder::TargetFinder::makeSendableData(int targetFound, double x, double y, double power, double groundDistance) {
	// Will be TargetFound, xTurn, PitchColor, RPMColor, GroundDistanceColor
	std::string data = std::to_string(targetFound) + "," + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(power) + "," + std::to_string(groundDistance);
	return data;
}
