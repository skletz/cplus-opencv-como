#include "quantifier.h"

quantifier::quantifier(){ }

quantifier::~quantifier(){ }

void quantifier::quantify(cv::Mat& descriptor, cv::Mat& output)
{
	int size = descriptor.cols;
	std::vector<uint8_t> result = std::vector<uint8_t>(size);
	output.reserve(mTEXTUREHISTSIZE * mFUZZYHISTSIZE);

	double min, temp;

	for(int iTexture = 0; iTexture < mTEXTUREHISTSIZE; iTexture++)
	{

		for (int iFuzzy = (iTexture * mFUZZYHISTSIZE); iFuzzy <  ((iTexture + 1) * mFUZZYHISTSIZE); iFuzzy++)
		{

			min = std::abs(descriptor.at<float>(iFuzzy) - mQuantizationTable[iTexture][0]);
			result[iFuzzy] = 0;

			for (uint8_t j = 1; ((j < 8) && (min != 0.0)); j++)
			{
				temp = std::abs(descriptor.at<float>(iFuzzy) - mQuantizationTable[iTexture][j]);
				if (temp < min)
				{
					min = temp;
					result[iFuzzy] = j;
				}
			}
		}

	}
	cv::Mat out(1, result.size(), CV_8UC1, result.data());
	out.copyTo(output);
}
