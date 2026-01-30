using System.Diagnostics;

public class FpsCounter
{
    private Stopwatch sw = new();
    private string[] nums = new string[200];

    private string fps = "";
    private float time = 0f;

    public FpsCounter()
    {
        for (int i = 0; i < 200; i++)
            nums[i] = i.ToString();
    }

    public string FPS => fps;

    public void Start() => sw.Restart();

    public void Stop()
    {
        sw.Stop();
        time += sw.ElapsedMilliseconds * 0.001f;

        if (time > 0.5f)
        {
            time = 0f;
            fps = nums[Math.Clamp(1000 / sw.ElapsedMilliseconds, 0, 199)];
        }
    }
}
