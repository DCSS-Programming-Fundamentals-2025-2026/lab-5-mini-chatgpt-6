public class NGramModel
{
    private float[][] _probs;
    private NGramCounts _counts;

    public NGramModel(int vocabSize){
        _probs = new float[vocabSize][];

        for(int i = 0; i < vocabSize; i++)
        {
            _probs[i] = new float[vocabSize];
        }

        _counts = new NGramCounts(vocabSize);
    }

    public void Train(ReadOnlySpan<int> tokens)
    {
        _counts.CountBigrams(tokens);

        for(int i = 0; i < _probs.Length; i++)
        {
            float rowSum = _counts.BigramCounts[i].Sum();

            if(rowSum > 0)
            {
                for(int j = 0; j < _probs[i].Length; j++)
                {
                    _probs[i][j] = _counts.BigramCounts[i][j] / rowSum;
                }
            }
        }
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        if(context.IsEmpty)
        {
            return new float[_probs.Length];
        }

        int last = context[context.Length-1];

        return _probs[last];
    }
}