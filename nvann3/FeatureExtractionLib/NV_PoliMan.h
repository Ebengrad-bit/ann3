#pragma once
#include "PolynomialManager.h"
class NV_PoliMan :
	public fe::PolynomialManager
{
public:
	NV_PoliMan();
	~NV_PoliMan();
	fe::ComplexMoments Decompose(cv::Mat blob);

	/**
	 * ¬осстановить картинку из разложени€.
	 * @param decomposition - разложение картинки в р€д.
	 * @return восстановленное изображение, имеет тип CV_64FC1.
	 */
	cv::Mat Recovery(fe::ComplexMoments& decomposition);

	/**
	 * ѕроинициализировать базис ортогональных полиномов ~ exp(jm*fi).
	 * @param n_max - максимальный радиальный пор€док полиномов.
	 * @param diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
	 */
	void InitBasis(int n_max, int diameter);

	/**
	 * ѕолучить описание объекта дл€ работы с полиномами.
	 * @return - описание объекта.
	 */
	std::string GetType();
};

