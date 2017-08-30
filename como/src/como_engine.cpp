#include "como_engine.h"

como_engine::como_engine()
{ }

como_engine::~como_engine()
{ }

//Basically, what the constructor should include
bool como_engine::allocate()
{
	//if failure return false

	return true;
}

//Basically, what the destructor should include
void como_engine::deallocate()
{

}

//Detect feature points in the image (image_in). Detected feature points are stored in featurePnts
bool como_engine::detect(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts)
{

	//if failure return false

	return true;
}

//Describe detected feature points (featurePnts) from the input image (image_in). Feature point descriptors are stored in the descriptors matrix (each row containing one descriptor)
bool como_engine::describe(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts, cv::Mat &descriptors)
{
	//no feature points, use all image points 
	if (featurePnts.empty())
	{
		return extract(image_in, descriptors);
	}

	//if failure return false
	return true;
}

bool como_engine::extract(cv::Mat& image, cv::Mat& features)
{

	return true;
}

bool como_engine::extractBlock(cv::Mat& image, cv::Rect& rectangle, cv::Mat& features)
{
	if (!image.data || image.empty())
	{
		std::cerr << "Error" << std::endl;
		return false;
	}
		

	int width = rectangle.x + rectangle.width;
	int height = rectangle.x + rectangle.height;

	for(int x = 0; x < width; x++)
	{
		for(int y = 0; y < height; y++)
		{
			
		}
	}

	return true;
}
