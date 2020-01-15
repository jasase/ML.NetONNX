using Microsoft.ML;
using Microsoft.ML.Data;
using ONNX.Test.model;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ONNX.Test
{
    class Program
    {

        //NETRON Tool for view ONNX

        static void Main(string[] args)
        {
            const string INPUT_COLUMN = "input_2";
            const string OUTPUT_COLUMN = "fc1000";

            Console.WriteLine("Hello World!");

            var context = new MLContext();

            var testData = context.Data.LoadFromEnumerable(new[] { new InputData { ImagePath = @"images\elephant.jpg" } });
            var fitData = context.Data.LoadFromEnumerable(new List<InputData>());
            var p1 = context.Transforms.LoadImages(outputColumnName: INPUT_COLUMN, imageFolder: "", inputColumnName: "ImagePath")
                                      .Append(context.Transforms.ResizeImages(outputColumnName: INPUT_COLUMN, imageWidth: 224, imageHeight: 224, inputColumnName: INPUT_COLUMN))
                                      .Append(context.Transforms.ExtractPixels(outputColumnName: INPUT_COLUMN,
                                                                               inputColumnName: INPUT_COLUMN,
                                                                               orderOfExtraction: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder.ABGR));

            var t = p1.Fit(fitData);
            var t2 = t.Transform(testData);

            var p2 = p1.Append(context.Transforms.ApplyOnnxModel(modelFile: "model\\model.onnx", outputColumnNames: new[] { OUTPUT_COLUMN }, inputColumnNames: new[] { INPUT_COLUMN }));
            var model = p2.Fit(fitData);


            IDataView scoredData = model.Transform(testData);
            var probabilities = scoredData.GetColumn<float[]>(OUTPUT_COLUMN).ToArray();

            foreach (var run in probabilities)
            {
                var max = run.Max();
                var avg = run.Average();

                Console.WriteLine("Max: " + max);
                Console.WriteLine("Avg: " + avg);

                for (int i = 0; i < run.Length; i++)
                {
                    if (run[i] > max - 5)
                    {
                        Console.WriteLine(run[i] + "\t" + ResultClasses.Data[i]);
                    }
                }
            }
        }
    }

    public class InputData
    {
        [LoadColumn(0)]
        public string ImagePath;

        //[LoadColumn(1)]
        //public string Label;
    }
}
