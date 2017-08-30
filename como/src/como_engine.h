#ifndef _COMO_ENGINE_H_
#define _COMO_ENGINE_H_

#include <opencv2/opencv.hpp>

class como_engine{

private:

	static int const mHISTSIZE = 144;
	static int const mBLOCKSIZE_MAX = 40;
	static int const mBLOCKSIZE_MIN = 20;

	std::vector<cv::Point2f> featurePnts;

	cv::Mat descriptors;

public:

	/**
	 * \brief 
	 */
	como_engine();

	/**
	 * \brief 
	 */
	~como_engine();

	/**
	 * \brief 
	 * \return 
	 */
	bool allocate();

	/**
	 * \brief 
	 * \param image_in 
	 * \param featurePnts 
	 * \return 
	 */
	bool detect(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts);

	/**
	 * \brief 
	 * \param image_in 
	 * \param featurePnts 
	 * \param descriptors 
	 * \return 
	 */
	bool describe(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts, cv::Mat &descriptors);

	/**
	 * \brief 
	 * \param image_in 
	 * \param descriptors 
	 * \return 
	 */
	bool extract(cv::Mat &image, cv::Mat &features);

	bool extractBlock(cv::Mat& image, cv::Rect& rectangle, cv::Mat& features);

	/**
	 * \brief 
	 */
	void deallocate();

};

#endif //_COMO_ENGINE_H_