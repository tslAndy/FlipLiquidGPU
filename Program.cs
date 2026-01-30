using System.Numerics;
using Raylib_cs;

public class Program
{
    private const int SCREEN_WIDTH = 1024;
    private const int SCREEN_HEIGHT = 1024;
    private const int SCALE = SCREEN_WIDTH / Game.WIDTH;

    public static void Main()
    {
        FpsCounter counter = new();
        Game game = new();

        Raylib.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Hello world");
        Texture2D texture = Raylib.LoadTexture("./circle.png");

        float texture_scale = Game.RADIUS / ((float)texture.Width * 0.5f) * SCALE;

        Thread.Sleep(5000);
        while (!Raylib.WindowShouldClose())
        {
            Raylib.BeginDrawing();
            Raylib.ClearBackground(Color.Black);

            counter.Start();
            game.Update();
            counter.Stop();

            for (int i = 0; i < Game.POINTS_COUNT; i++)
            {
                Game.Point point = game.GetPoint(i);
                point.x *= SCALE;
                point.y *= SCALE;
                Raylib.DrawTextureEx(
                    texture,
                    new Vector2(point.x - SCALE / 2, SCREEN_HEIGHT - point.y - SCALE / 2),
                    0f,
                    texture_scale,
                    Color.Blue
                );
            }

            Raylib.DrawText(counter.FPS, 20, 20, 30, Color.Green);

            Raylib.EndDrawing();
        }

        game.Dispose();

        Raylib.UnloadTexture(texture);
        Raylib.CloseWindow();
    }
}
