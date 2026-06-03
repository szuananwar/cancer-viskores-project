#include <viskores/cont/Initialize.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/filter/contour/Contour.h>

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[])
{
    viskores::cont::Initialize(argc, argv);

    if (argc < 4)
    {
        std::cerr << "Usage: ./ExtractTumorFeatures input.vti class_label output.csv\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string classLabel = argv[2];
    std::string outputCsv = argv[3];

    try
    {
        viskores::io::VTKDataSetReader reader(inputFile);
        viskores::cont::DataSet dataSet = reader.ReadDataSet();

        viskores::filter::contour::Contour contour;
        contour.SetActiveField("MRI");
        contour.SetIsoValue(0, 0.5);

        viskores::cont::DataSet isoSurface = contour.Execute(dataSet);

        auto numberOfPoints = isoSurface.GetNumberOfPoints();
        auto numberOfCells = isoSurface.GetNumberOfCells();

        std::ofstream out(outputCsv, std::ios::app);

        out << inputFile << ","
            << classLabel << ","
            << numberOfPoints << ","
            << numberOfCells << "\n";

        out.close();

        std::cout << "Processed: " << inputFile << std::endl;
        std::cout << "Class: " << classLabel << std::endl;
        std::cout << "IsoSurface Points: " << numberOfPoints << std::endl;
        std::cout << "IsoSurface Cells: " << numberOfCells << std::endl;
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
