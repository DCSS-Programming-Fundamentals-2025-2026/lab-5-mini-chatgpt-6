using NGram.ModelFactory;

Console.WriteLine("Hello, World!");

NGramModelFactory factory = new NGramModelFactory();
NGramModel model = factory.CreateBigramModel(5);

int[] data = new int[] { 0, 1, 2, 1, 3 };
model.Train(data);

int[] context = new int[] { 1 };
float[] scores = model.NextTokenScores(context);

for (int i = 0; i < scores.Length; i++)
{
    Console.WriteLine($"{i}: {scores[i]}");
}
