//============================================================================
// Name        : COMO.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "como_engine.h"
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {

	std::cout << "COMO descriptor (Color moments)" << std::endl;

	if (argc < 3)
	{
		std::cout << "Demo of COMO descriptor - Too few arguments:" << std::endl;
		std::cout << "\t arg1: Image File" << std::endl;
		std::cout << "\t arg2: Output File" << std::endl;

		std::getchar();

		return EXIT_FAILURE;
	}

	std::string videoFile = argv[1];
	std::string outputFile = argv[2];

	cv::Mat image = cv::imread(videoFile, CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		std::cout << "Could not open or find the image: " << argv[1] << std::endl;
		return -1;
	}

	como_engine comoExtractor;
	comoExtractor.allocate();

	//while (inputImageStream)
	//{
		//std::vector<cv::Point2f> featurePnts;
	//	cv::Mat &descriptors;

	//	image = get_newImage(); // no need to be implemented.
		//comoExtractor.detect(image, featurePnts);
	//	comoExtractor.describe(image, featurePnts, descriptors);
	//}

	cv::Mat descriptors;
	comoExtractor.describe(image, descriptors);

	comoExtractor.deallocate();

}
