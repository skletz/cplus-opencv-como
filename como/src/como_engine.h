#ifndef _COMO_ENGINE_H_
#define _COMO_ENGINE_H_

#include <opencv2/opencv.hpp>
#include "fuzzificator.h"
#include "quantifier.h"

class como_engine{

private:

	static int const mCOMOSIZE = 144;
	static int const mBLOCKSIZE_MAX = 40;
	static int const mBLOCKSIZE_MIN = 20;
	static int const mGRAYHISTBINS = 256;
	static int const mNRTEXTURES = 6;
	static int const mNRHUMOMETS = 7;

	double mTextureDefinitionTable[mNRTEXTURES][mNRHUMOMETS] = {
		{ 0.0012801877159444332, 1.2811997974150548E-8, 2.931517692794886E-11, 2.3361585476863737E-11, 1.0229383520567805E-21, -4.741043401863473E-16, 3.489450463009515E-23 },
		{ 0.030098519274221386, 9.553200793114547E-6, 2.7312098265223947E-7, 1.5846418079328335E-6, -2.6894242032236695E-12, 5.431901588135399E-9, -1.6868751727422065E-12 },
		{ 0.0018139530164278453, 5.941872634495257E-8, 1.4200444288697612E-10, 1.5023611260922216E-10, 1.3373151439243982E-20, 1.1760743231129731E-14, 1.3487898261349881E-20 },
		{ 0.004440612110292505, 6.320931310602498E-7, 5.4004430112564E-9, 1.0423679617819742E-8, -2.7085204326667915E-16, 1.7090726156377653E-11, 9.51706439192856E-16 },
		{ 9.639630464555355E-4, 7.859161865855281E-9, 1.1230404627214495E-11, 8.291087328844257E-12, 1.372616300158933E-22, -4.3513261260527616E-16, -3.862637133906026E-24 },
		{ 0.10135058748498203, 4.290771163627242E-5, 6.178005148446455E-6, 1.9548008476574724E-5, 4.4473620282147234E-11, 1.6752667650020017E-7, 2.758163163459048E-11 }
	};

	double mThresholdsTable[6] = {
		0.005, 0.005, 0.005, 0.0195, 0.15, 0.99
	};

	fuzzificator* mFuzzificator;

	quantifier* mQuantifier;

public:

	/**
	 * \brief Constructor
	 */
	como_engine();

	/**
	 * \brief Destructor
	 */
	~como_engine();

	/**
	 * \brief The function include what the constructor basically include
	 * \return false on failure
	 */
	bool allocate();

	/**
	* \brief The function include what the destructor basically include
	*/
	void deallocate();

	/**
	 * \brief The function randomly creates feature points. The number of feature points depends on the size of image_in: mNRBLOCKS > (rows * cols) ? mNRBLOCKS : (rows * cols).
	 * 
	 * \param image_in 8-bit input image
	 * \param featurePnts vector of points to store the feature points
	 * \return false on failure 
	 */
	bool detect(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts);

	/**
	 * \brief Describe detected feature points (featurePnts) from the input image (image_in). Feature point descriptors are stored in the descriptors matrix (each row containing one descriptor)
	 * \param image_in 
	 * \param featurePnts 
	 * \param descriptors 
	 * \return false on failure 
	 */
	bool describe(cv::Mat &image_in, std::vector<cv::Point2f> &featurePnts, cv::Mat &descriptors);

	/**
	 * \brief Describes each block of an image. The image is subdivided into 40*40 blocks (max) or 20*20 blocks (min). If the block contains enough texture information, the color (HSV) as well as texture (HuMoments) are calculated and summed up in the descriptor.
	 * \param image 8-bit input image
	 * \param descriptor output como descriptor
	 * \return false on failure 
	 */
	bool describe(cv::Mat &image, cv::Mat &descriptor);

	/**
	 * \brief Extract como features of a block
	 * \param imageBlock input block of an image
	 * \param features 
	 * \return false on failure 
	 */
	bool extractFromBlock(cv::Mat &imageBlock, cv::Mat &features);

	void convertToGrayscale(cv::Mat& image, cv::Mat& gray);

	void calculateGrayscaleHistogram(cv::Mat& gray, cv::Mat& hist);

	void calculateHuMoments(cv::Mat& gray, cv::Mat& huMoments);

	float calculateEntropy(cv::Mat& grayscaleHist, int area);

	void calculateDistances(cv::Mat& huMoments, cv::Mat& huTable, std::vector<double>& distances);

	void showHistogram(std::string name, cv::MatND &hist, int height, int width);

	void showImage(std::string name, cv::MatND &image);

	bool test(cv::Mat &image, cv::Mat &descriptor);

};

#endif //_COMO_ENGINE_H_