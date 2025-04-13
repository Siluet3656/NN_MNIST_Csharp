using System;
using System.IO;

public static class MNISTReader
{
    private static string BaseDataPath => Path.Combine(Directory.GetCurrentDirectory(), "Data");

    public static (double[], int)[] LoadDataSet(string imagesFile, string labelsFile, int maxItems = int.MaxValue)
    {
        string imagesPath = Path.Combine(BaseDataPath, imagesFile);
        string labelsPath = Path.Combine(BaseDataPath, labelsFile);

        ValidateFile(imagesPath);
        ValidateFile(labelsPath);

        using var imagesStream = new FileStream(imagesPath, FileMode.Open);
        using var labelsStream = new FileStream(labelsPath, FileMode.Open);
        using var imagesReader = new BinaryReader(imagesStream);
        using var labelsReader = new BinaryReader(labelsStream);

        imagesReader.ReadInt32();
        int imageCount = ReadBigEndianInt32(imagesReader);
        int rows = ReadBigEndianInt32(imagesReader);
        int cols = ReadBigEndianInt32(imagesReader);

        labelsReader.ReadInt32();
        int labelCount = ReadBigEndianInt32(labelsReader);

        int itemsToRead = Math.Min(Math.Min(imageCount, labelCount), maxItems);
        var dataset = new (double[], int)[itemsToRead];

        for (int i = 0; i < itemsToRead; i++)
        {
            byte[] pixels = imagesReader.ReadBytes(rows * cols);
            double[] imageData = pixels.Select(p => p / 255.0).ToArray();
            byte label = labelsReader.ReadByte();
            dataset[i] = (imageData, label);
        }

        return dataset;
    }

    private static void ValidateFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"MNIST file not found: {path}");
    }

    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}