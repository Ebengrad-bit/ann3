#pragma once
#include "IBlobProcessor.h"
class NV_IBlob :
	public fe::IBlobProcessor
{
public:
	NV_IBlob();
	~NV_IBlob();
	std::string GetType();
	std::vector<cv::Mat> DetectBlobs(cv::Mat image);

	/**
	 * Привести размер смежных областей к единому масштабу.
	 * @param blobs - смежные области, должны представлять собой белые символы на чёрном фоне, тип CV_8UC1.
	 * @param normilized_blobs - буфер для записи смежных областей единого размера.
	 *               Смежные области после записи представляют собой белые буквы на чёрном фоне, тип CV_8UC1.
	 * @param side - сторона квадрата на котором будет отрисована нормализованная смежная область.
	 */
	std::vector<cv::Mat> NormalizeBlobs(
		std::vector<cv::Mat>& blobs,
		int side
	);
};

