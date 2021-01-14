#define FEATURE_DLL_EXPORTS

#include "FeatureExtraction.h"

#include "NV_IBlob.h"
#include "NV_PoliMan.h"


std::string fe::GetTestString()
{
	return "fe by NV";
}

/**
 * Создать обработчик смежных областей.
 * @return обработчик смежных областей.
 */
std::shared_ptr<fe::IBlobProcessor> fe::CreateBlobProcessor() 
{
	return std::shared_ptr<fe::IBlobProcessor>(new NV_IBlob());
}

/**
 * Создать объект, ответственный за работу с полиномами.
 * @return объект, ответсвенный за работу с полиномами.
 */
std::shared_ptr<fe::PolynomialManager> fe::CreatePolynomialManager()
{
	return std::shared_ptr<fe::PolynomialManager>(new NV_PoliMan());
}

