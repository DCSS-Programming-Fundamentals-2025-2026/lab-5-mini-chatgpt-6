namespace NGram;
public class TrigramModel
{
    private Dictionary<(int, int), float[]> _trigramProbs { get; set; }
    private float[][] _bigramProbs { get; set; }
    private NGramCounts _counts;

    public TrigramModel(int vocabSize)
    {
        _trigramProbs = new Dictionary<(int, int), float[]>();
        _bigramProbs = new float[vocabSize][];
        for (int i = 0; i < vocabSize; i++)
        {
            _bigramProbs[i] = new float[vocabSize];
        }

        _counts = new NGramCounts(vocabSize);
    }

    public void Train(ReadOnlySpan<int> tokens)
    {
        _counts.CountTrigrams(tokens);
        _counts.CountBigrams(tokens);

        for (int i = 0; i < _bigramProbs.Length; i++)
        {
            float rowSum = _counts.BigramCounts[i].Sum();

            if (rowSum > 0)
            {
                for (int j = 0; j < _bigramProbs[i].Length; j++)
                {
                    _bigramProbs[i][j] = _counts.BigramCounts[i][j] / rowSum;
                }
            }
        }

        foreach (var item in _counts.trigramCounts)
        {
            float rowSum = item.Value.Sum();

            if (rowSum > 0)
            {
                float[] normalized = new float[item.Value.Length];
                for (int i = 0; i < item.Value.Length; i++)
                {
                    normalized[i] = item.Value[i] / rowSum;
                }
                _trigramProbs[item.Key] = normalized;
            }
        }
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        if (context.IsEmpty)
        {
            return new float[_bigramProbs.Length];
        }

        if (context.Length >= 2)
        {
            int p2 = context[context.Length - 2];
            int p1 = context[context.Length - 1];
            (int, int) key = (p2, p1);

            if (_trigramProbs.TryGetValue(key, out float[] trigramScores))
            {
                return trigramScores;
            }
        }

        int last = context[context.Length - 1];
        return _bigramProbs[last];
    }
}
