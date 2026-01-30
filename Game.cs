using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using ILGPU.Algorithms;

public partial class Game : IDisposable
{
    public const int WIDTH = 128;
    public const int HEIGHT = 128;

    public const int GRID_SIZE = 1;
    public const int GRID_WIDTH = WIDTH / GRID_SIZE;
    public const int GRID_HEIGHT = HEIGHT / GRID_SIZE;
    private const float INV_GRID_SIZE = 1f / GRID_SIZE;

    private const int CHUNK_SIZE = 2;
    private const int CHUNKS_X = WIDTH / CHUNK_SIZE;
    private const int CHUNKS_Y = HEIGHT / CHUNK_SIZE;
    private const float INV_CHUNK_SIZE = 1f / CHUNK_SIZE;

    public const int POINTS_COUNT = 10000;
    public const float RADIUS = 0.5f;
    private const float DT = 0.016f;
    private const float G = -9.81f;
    private const float OVERRELAX = 1.9f;
    private const float STIFFNESS = 1.0f;
    private const float FLIP_RATIO = 0.9f;
    private const float SLOPE = 0.05f;
    private const int COLLISON_SOLVE_ITERATIONS = 1;
    private const int GRID_SOLVE_ITERATIONS = 20;

    private Context context;
    private Accelerator accelerator;

    private MemoryBuffer1D<Point, Stride1D.Dense> main_points, swap_points;
    private MemoryBuffer1D<int, Stride1D.Dense> buckets_used, buckets_total;

    private MemoryBuffer2D<float, Stride2D.DenseX> densities, densities_total;
    private MemoryBuffer2D<int, Stride2D.DenseX> fluid_cells, fluid_cells_total;

    private MemoryBuffer2D<float, Stride2D.DenseX> grid_main_vx, grid_old_vx, grid_delta_vx;
    private MemoryBuffer2D<float, Stride2D.DenseX> grid_main_vy, grid_old_vy, grid_delta_vy;

    private Action<Index1D, ArrayView<Point>> move, bound;
    private Action<Index1D, ArrayView<Point>, ArrayView<int>> count_points;
    private Action<Index1D, ArrayView<int>, int> sweep_up_points, sweep_down_points;
    private Action<Index1D, ArrayView<Point>, ArrayView<Point>, ArrayView<int>, ArrayView<int>> sort_points;
    private Action<Index1D, ArrayView<Point>, ArrayView<Point>, ArrayView<int>> solve_collisions;
    private Action<Index1D, ArrayView<Point>, ArrayView2D<int, Stride2D.DenseX>> fill_fluid_cells;
    private Action<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>> fill_densities;
    private Action<Index2D, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> mask_densities;
    private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, int> sweep_up_densities;
    private Action<Index1D, ArrayView2D<int, Stride2D.DenseX>, int> sweep_up_fluid_cells;
    private Action<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> transfer_to_grid;
    private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> divide_vel_by_delta;
    private Action<Index2D, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, int> solve_pressure;
    private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> bound_wall_vx, bound_wall_vy;
    private Action<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, int> transfer_to_point;

    private float average_density;
    private int[,] fluid_cells_count = new int[1, 1];
    private float[,] fluid_cells_density = new float[1, 1];

    private Point[] local_points = new Point[POINTS_COUNT];

    public Game()
    {
        context = Context.Create(builder => builder.OpenCL());
        accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        main_points = accelerator.Allocate1D<Point>(POINTS_COUNT);
        swap_points = accelerator.Allocate1D<Point>(POINTS_COUNT);

        buckets_used = accelerator.Allocate1D<int>(CHUNKS_X * CHUNKS_Y + 1);
        buckets_total = accelerator.Allocate1D<int>(CHUNKS_X * CHUNKS_Y + 1);

        densities = accelerator.Allocate2DDenseX<float>((GRID_WIDTH, GRID_HEIGHT));
        densities_total = accelerator.Allocate2DDenseX<float>((GRID_WIDTH, GRID_HEIGHT));

        fluid_cells = accelerator.Allocate2DDenseX<int>((GRID_WIDTH, GRID_HEIGHT));
        fluid_cells_total = accelerator.Allocate2DDenseX<int>((GRID_WIDTH, GRID_HEIGHT));

        grid_main_vx = accelerator.Allocate2DDenseX<float>((GRID_WIDTH + 1, GRID_HEIGHT));
        grid_old_vx = accelerator.Allocate2DDenseX<float>((GRID_WIDTH + 1, GRID_HEIGHT));
        grid_delta_vx = accelerator.Allocate2DDenseX<float>((GRID_WIDTH + 1, GRID_HEIGHT));

        grid_main_vy = accelerator.Allocate2DDenseX<float>((GRID_WIDTH, GRID_HEIGHT + 1));
        grid_old_vy = accelerator.Allocate2DDenseX<float>((GRID_WIDTH, GRID_HEIGHT + 1));
        grid_delta_vy = accelerator.Allocate2DDenseX<float>((GRID_WIDTH, GRID_HEIGHT + 1));

        move = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>>(Move);
        bound = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>>(Bound);

        count_points = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView<int>>(Count_Points);
        sweep_up_points = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(Sweep_Up_Points);
        sweep_down_points = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(Sweep_Down_Points);
        sort_points = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView<Point>, ArrayView<int>, ArrayView<int>>(Sort_Points);

        solve_collisions = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView<Point>, ArrayView<int>>(Solve_Collisions);

        fill_fluid_cells = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView2D<int, Stride2D.DenseX>>(Fill_Fluid_Cells);
        fill_densities = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>>(Fill_Densities);

        mask_densities = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(Mask_Density);
        sweep_up_densities = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, int>(Sweep_Up_Densities);
        sweep_up_fluid_cells = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<int, Stride2D.DenseX>, int>(Sweep_Up_Fluid_Cells);
        divide_vel_by_delta = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(Divide_Vel_By_Delta);
        transfer_to_grid = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(Transfer_To_Grid);

        solve_pressure = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, int>(Solve_Pressure);
        bound_wall_vx = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(Bound_Wall_VX);
        bound_wall_vy = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(Bound_Wall_VY);
        transfer_to_point = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Point>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, int>(Transfer_To_Point);
        Init();
    }

    public void Update()
    {
        DispatchMoving();
        DispatchDensityCount();
        DispatchTransferToGrid();
        DispatchGridSolving();
        DispatchTransferToPoints();

        main_points.CopyToCPU(local_points);
    }

    public Point GetPoint(int index) => local_points[index];

    #region Dispatchers

    private void DispatchMoving()
    {
        move(POINTS_COUNT, main_points.View);
        accelerator.Synchronize();

        bound(POINTS_COUNT, main_points.View);
        accelerator.Synchronize();

        int chunks_amount = CHUNKS_X * CHUNKS_Y;

        buckets_used.MemSetToZero();
        buckets_total.MemSet(accelerator.DefaultStream, 0, 0, 4 * chunks_amount);

        count_points(POINTS_COUNT, main_points.View, buckets_total.View);
        accelerator.Synchronize();

        for (int step = 2; step <= chunks_amount; step *= 2)
        {
            sweep_up_points(chunks_amount / step, buckets_total.View, step);
            accelerator.Synchronize();
        }

        for (int step = chunks_amount; step >= 2; step /= 2)
        {
            sweep_down_points(chunks_amount / step, buckets_total.View, step);
            accelerator.Synchronize();
        }

        sort_points(POINTS_COUNT, main_points.View, swap_points.View, buckets_used.View, buckets_total.View);
        accelerator.Synchronize();
        (main_points, swap_points) = (swap_points, main_points);

        for (int i = 0; i < COLLISON_SOLVE_ITERATIONS; i++)
        {
            solve_collisions(POINTS_COUNT, main_points.View, swap_points.View, buckets_total.View);
            accelerator.Synchronize();
            (main_points, swap_points) = (swap_points, main_points);

            bound(POINTS_COUNT, main_points.View);
            accelerator.Synchronize();
        }
    }

    private void DispatchDensityCount()
    {
        fluid_cells.MemSetToZero();
        fill_fluid_cells(POINTS_COUNT, main_points.View, fluid_cells.View);
        accelerator.Synchronize();

        densities.MemSetToZero();
        fill_densities(POINTS_COUNT, main_points.View, densities.View);
        accelerator.Synchronize();

        mask_densities((GRID_WIDTH, GRID_HEIGHT), fluid_cells.View, densities.View);
        accelerator.Synchronize();

        densities_total.CopyFrom(densities);
        fluid_cells_total.CopyFrom(fluid_cells);

        int cells_amount = GRID_WIDTH * GRID_HEIGHT;
        for (int step = 2; step <= cells_amount; step *= 2)
        {
            sweep_up_densities(cells_amount / step, densities_total.View, step);
            accelerator.Synchronize();
        }

        for (int step = 2; step <= cells_amount; step *= 2)
        {
            sweep_up_fluid_cells(cells_amount / step, fluid_cells_total.View, step);
            accelerator.Synchronize();
        }

        densities_total.View.SubView((GRID_WIDTH - 1, GRID_HEIGHT - 1), (1, 1)).CopyToCPU(fluid_cells_density);
        fluid_cells_total.View.SubView((GRID_WIDTH - 1, GRID_HEIGHT - 1), (1, 1)).CopyToCPU(fluid_cells_count);
        average_density = fluid_cells_density[0, 0] / fluid_cells_count[0, 0];
    }

    private void DispatchTransferToGrid()
    {
        grid_main_vx.MemSetToZero();
        grid_delta_vx.MemSetToZero();
        transfer_to_grid(POINTS_COUNT, main_points.View, grid_main_vx.View, grid_delta_vx.View, 1);
        accelerator.Synchronize();
        divide_vel_by_delta((GRID_WIDTH - 2, GRID_HEIGHT - 2), grid_main_vx.View, grid_delta_vx.View);
        accelerator.Synchronize();
        grid_old_vx.CopyFrom(grid_main_vx);

        grid_main_vy.MemSetToZero();
        grid_delta_vy.MemSetToZero();
        transfer_to_grid(POINTS_COUNT, main_points.View, grid_main_vy.View, grid_delta_vy.View, 0);
        accelerator.Synchronize();
        divide_vel_by_delta((GRID_WIDTH - 2, GRID_HEIGHT - 2), grid_main_vy.View, grid_delta_vy.View);
        accelerator.Synchronize();
        grid_old_vy.CopyFrom(grid_main_vy);
    }

    private void DispatchGridSolving()
    {
        for (int i = 0; i < GRID_SOLVE_ITERATIONS; i++)
        {
            Index2D dispatch = ((GRID_WIDTH - 2) / 2, GRID_HEIGHT - 2);
            solve_pressure(dispatch, fluid_cells.View, grid_main_vx.View, grid_main_vy.View, densities.View, average_density, 0);
            accelerator.Synchronize();

            solve_pressure(dispatch, fluid_cells.View, grid_main_vx.View, grid_main_vy.View, densities.View, average_density, 1);
            accelerator.Synchronize();

            bound_wall_vx((2, GRID_HEIGHT - 2), grid_main_vx.View);
            accelerator.Synchronize();

            bound_wall_vy((GRID_WIDTH - 2, 2), grid_main_vy.View);
            accelerator.Synchronize();
        }
    }

    private void DispatchTransferToPoints()
    {
        transfer_to_point(POINTS_COUNT, main_points.View, grid_main_vx.View, grid_old_vx.View, fluid_cells.View, 1);
        accelerator.Synchronize();

        transfer_to_point(POINTS_COUNT, main_points.View, grid_main_vy.View, grid_old_vy.View, fluid_cells.View, 0);
        accelerator.Synchronize();
    }
    #endregion

    #region Kernels 
    private static void Move(Index1D ind, ArrayView<Point> points)
    {
        Point point = points[ind];
        point.vy += DT * G;
        point.x += DT * point.vx;
        point.y += DT * point.vy;
        points[ind] = point;
    }

    private static void Bound(Index1D ind, ArrayView<Point> points)
    {
        Point point = points[ind];
        if (point.x - RADIUS < 1f)
        {
            point.x = RADIUS + 1f;
            point.vx = 0f;
        }
        else if (point.x + RADIUS > WIDTH - 1f)
        {
            point.x = WIDTH - RADIUS - 1f;
            point.vx = 0f;
        }

        if (point.y - RADIUS < 1f)
        {
            point.y = RADIUS + 1f;
            point.vy = 0f;
        }
        else if (point.y + RADIUS > HEIGHT - 1f)
        {
            point.y = HEIGHT - RADIUS - 1f;
            point.vy = 0f;
        }
        points[ind] = point;
    }

    private static void Count_Points(
            Index1D ind,
            ArrayView<Point> points,
            ArrayView<int> buckets_total)
    {
        Point point = points[ind];
        int cx = (int)(point.x * INV_CHUNK_SIZE);
        int cy = (int)(point.y * INV_CHUNK_SIZE);
        int ci = cy * CHUNKS_X + cx;
        Atomic.Add(ref buckets_total[ci], 1);
    }

    private static void Sweep_Up_Points(
            Index1D ind,
            ArrayView<int> buckets_total,
            int step)
    {
        ind = ind * step + step - 1;
        buckets_total[ind] += buckets_total[ind - step / 2];
    }

    private static void Sweep_Down_Points(
            Index1D ind,
            ArrayView<int> buckets_total,
            int step)
    {
        ind = ind * step + step - 1;

        if (step == buckets_total.Length - 1)
            buckets_total[ind] = 0;

        int temp = buckets_total[ind];
        buckets_total[ind] += buckets_total[ind - step / 2];
        buckets_total[ind - step / 2] = temp;
    }

    private static void Sort_Points(
            Index1D ind,
            ArrayView<Point> main_points,
            ArrayView<Point> swap_points,
            ArrayView<int> used,
            ArrayView<int> total)
    {
        Point point = main_points[ind];
        int cx = (int)(point.x * INV_CHUNK_SIZE);
        int cy = (int)(point.y * INV_CHUNK_SIZE);
        int ci = cy * CHUNKS_X + cx;
        int swapIndex = total[ci] + Atomic.Add(ref used[ci], 1);
        swap_points[swapIndex] = point;
    }

    private static void Solve_Collisions(
            Index1D ind,
            ArrayView<Point> main_points,
            ArrayView<Point> swap_points,
            ArrayView<int> total)
    {
        Point point = main_points[ind];

        int cx = (int)(point.x * INV_CHUNK_SIZE);
        int cy = (int)(point.y * INV_CHUNK_SIZE);

        int startX = XMath.Max(cx - 1, 0);
        int endX = XMath.Min(cx + 2, CHUNKS_X);

        int startY = XMath.Max(cy - 1, 0);
        int endY = XMath.Min(cy + 2, CHUNKS_Y);

        float deltaX = 0, deltaY = 0;

        for (int ty = startY; ty < endY; ty++)
        {
            for (int tx = startX; tx < endX; tx++)
            {
                int chunkIndex = ty * CHUNKS_X + tx;
                int chunkStart = total[chunkIndex];
                int chunkEnd = total[chunkIndex + 1];

                for (int j = chunkStart; j < chunkEnd; j++)
                {
                    if (ind == j)
                        continue;

                    Point temp = main_points[j];
                    float nx = point.x - temp.x;
                    float ny = point.y - temp.y;
                    float distSqr = nx * nx + ny * ny;

                    if (distSqr > 4 * RADIUS * RADIUS)
                        continue;

                    float dist = XMath.Sqrt(distSqr);
                    nx /= dist;
                    ny /= dist;

                    float overlap = XMath.Max(RADIUS - 0.5f * dist - SLOPE, 0f);
                    deltaX += overlap * nx;
                    deltaY += overlap * ny;
                }
            }
        }

        point.x += deltaX;
        point.y += deltaY;
        swap_points[ind] = point;
    }

    private static void Fill_Fluid_Cells(
            Index1D ind,
            ArrayView<Point> points,
            ArrayView2D<int, Stride2D.DenseX> fluid_cell)
    {
        Point point = points[ind];
        point.x *= INV_GRID_SIZE;
        point.y *= INV_GRID_SIZE;
        int cell_x = (int)(point.x);
        int cell_y = (int)(point.y);
        fluid_cell[cell_x, cell_y] = 1;
    }

    private static void Fill_Densities(
            Index1D ind,
            ArrayView<Point> points,
            ArrayView2D<float, Stride2D.DenseX> densities)
    {
        Point point = points[ind];

        point.x *= INV_GRID_SIZE;
        point.y *= INV_GRID_SIZE;

        point.x -= 0.5f;
        point.y -= 0.5f;

        int cx = (int)(point.x);
        int cy = (int)(point.y);
        float dx = point.x - cx;
        float dy = point.y - cy;

        float w1 = (1 - dx) * (1 - dy);
        float w2 = dx * (1 - dy);
        float w3 = dx * dy;
        float w4 = (1 - dx) * dy;

        Atomic.Add(ref densities[cx, cy], w1);
        Atomic.Add(ref densities[cx + 1, cy], w2);
        Atomic.Add(ref densities[cx + 1, cy + 1], w3);
        Atomic.Add(ref densities[cx, cy + 1], w4);
    }

    private static void Mask_Density(
            Index2D ind,
            ArrayView2D<int, Stride2D.DenseX> fluid_cells,
            ArrayView2D<float, Stride2D.DenseX> densities)
    {
        densities[ind] *= fluid_cells[ind];
    }

    private static void Sweep_Up_Densities(
            Index1D ind,
            ArrayView2D<float, Stride2D.DenseX> densities_total,
            int step)
    {
        Index1D ind1 = step * (ind + 1) - 1;
        Index1D ind2 = ind1 - step / 2;

        int x1 = ind1 % GRID_WIDTH;
        int y1 = ind1 / GRID_WIDTH;

        int x2 = ind2 % GRID_WIDTH;
        int y2 = ind2 / GRID_WIDTH;

        densities_total[(x1, y1)] += densities_total[(x2, y2)];
    }

    private static void Sweep_Up_Fluid_Cells(
            Index1D ind,
            ArrayView2D<int, Stride2D.DenseX> fluid_cells_total,
            int step)
    {
        Index1D ind1 = step * (ind + 1) - 1;
        Index1D ind2 = ind1 - step / 2;

        int x1 = ind1 % GRID_WIDTH;
        int y1 = ind1 / GRID_WIDTH;

        int x2 = ind2 % GRID_WIDTH;
        int y2 = ind2 / GRID_WIDTH;

        fluid_cells_total[(x1, y1)] += fluid_cells_total[(x2, y2)];
    }

    private static void Transfer_To_Grid(
            Index1D ind,
            ArrayView<Point> points,
            ArrayView2D<float, Stride2D.DenseX> grid_main_vel,
            ArrayView2D<float, Stride2D.DenseX> grid_delta_vel,
            int transfer_vx)
    {
        Point point = points[ind];
        point.x *= INV_GRID_SIZE;
        point.y *= INV_GRID_SIZE;
        float vel;
        if (transfer_vx == 1)
        {
            point.y -= 0.5f;
            vel = point.vx;
        }
        else
        {
            point.x -= 0.5f;
            vel = point.vy;
        }

        int cx = (int)(point.x);
        int cy = (int)(point.y);

        float dx = point.x - cx;
        float dy = point.y - cy;

        float w1 = (1 - dx) * (1 - dy);
        float w2 = dx * (1 - dy);
        float w3 = dx * dy;
        float w4 = (1 - dx) * dy;

        Atomic.Add(ref grid_main_vel[cx, cy], w1 * vel);
        Atomic.Add(ref grid_main_vel[cx + 1, cy], w2 * vel);
        Atomic.Add(ref grid_main_vel[cx + 1, cy + 1], w3 * vel);
        Atomic.Add(ref grid_main_vel[cx, cy + 1], w4 * vel);

        Atomic.Add(ref grid_delta_vel[cx, cy], w1);
        Atomic.Add(ref grid_delta_vel[cx + 1, cy], w2);
        Atomic.Add(ref grid_delta_vel[cx + 1, cy + 1], w3);
        Atomic.Add(ref grid_delta_vel[cx, cy + 1], w4);
    }

    private static void Divide_Vel_By_Delta(
            Index2D ind,
            ArrayView2D<float, Stride2D.DenseX> vel,
            ArrayView2D<float, Stride2D.DenseX> delta)
    {
        ind += (1, 1);
        float d = delta[ind];
        if (d > 0.001f)
            vel[ind] /= d;
    }

    private static void Solve_Pressure(
            Index2D ind,
            ArrayView2D<int, Stride2D.DenseX> fluid_cell,
            ArrayView2D<float, Stride2D.DenseX> grid_main_vx,
            ArrayView2D<float, Stride2D.DenseX> grid_main_vy,
            ArrayView2D<float, Stride2D.DenseX> densities,
            float average_density,
            int baseOffset)
    {
        ind = new Index2D(((ind.Y + baseOffset) % 2) + ind.X * 2 + 1, ind.Y + 1);
        if (fluid_cell[ind] != 1)
            return;

        float vel_left = grid_main_vx[(ind.X, ind.Y)];
        float vel_right = grid_main_vx[(ind.X + 1, ind.Y)];
        float vel_up = grid_main_vy[(ind.X, ind.Y + 1)];
        float vel_down = grid_main_vy[(ind.X, ind.Y)];

        float div = (vel_right - vel_left + vel_up - vel_down) * 0.25f * OVERRELAX;
        float drift_compensate = (densities[ind] - average_density) * STIFFNESS;
        if (drift_compensate > 0f)
            div -= drift_compensate;

        grid_main_vx[(ind.X, ind.Y)] += div;
        grid_main_vx[(ind.X + 1, ind.Y)] -= div;
        grid_main_vy[(ind.X, ind.Y + 1)] -= div;
        grid_main_vy[(ind.X, ind.Y)] += div;
    }

    private static void Bound_Wall_VX(Index2D ind, ArrayView2D<float, Stride2D.DenseX> grid_vx)
    {
        int x = 1 + ind.X * (GRID_WIDTH - 2);
        int y = 1 + ind.Y;
        int dx = 2 * (1 - ind.X) - 1;
        grid_vx[(x, y)] = 0;
    }

    private static void Bound_Wall_VY(Index2D ind, ArrayView2D<float, Stride2D.DenseX> grid_vy)
    {
        int x = 1 + ind.X;
        int y = 1 + ind.Y * (GRID_HEIGHT - 2);
        int dy = 2 * (1 - ind.Y) - 1;
        grid_vy[(x, y)] = 0;
    }

    private static void Transfer_To_Point(
            Index1D ind,
            ArrayView<Point> points,
            ArrayView2D<float, Stride2D.DenseX> grid_main,
            ArrayView2D<float, Stride2D.DenseX> grid_old,
            ArrayView2D<int, Stride2D.DenseX> fluid_cell,
            int transfer_vx)
    {
        Point point = points[ind];

        float tx = point.x * INV_GRID_SIZE;
        float ty = point.y * INV_GRID_SIZE;

        Index2D offset;
        if (transfer_vx == 1)
        {
            ty -= 0.5f;
            offset = (-1, 0);
        }
        else
        {
            tx -= 0.5f;
            offset = (0, -1);
        }
        int cx = (int)(tx);
        int cy = (int)(ty);

        float dx = tx - cx;
        float dy = ty - cy;

        float w1 = (1 - dx) * (1 - dy) * (fluid_cell[(cx, cy)] | fluid_cell[(cx, cy) + offset]);
        float w2 = dx * (1 - dy) * (fluid_cell[(cx + 1, cy)] | fluid_cell[(cx + 1, cy) + offset]);
        float w3 = dx * dy * (fluid_cell[(cx + 1, cy + 1)] | fluid_cell[(cx + 1, cy + 1) + offset]);
        float w4 = (1 - dx) * dy * (fluid_cell[(cx, cy + 1)] | fluid_cell[(cx, cy + 1) + offset]);

        float d = w1 + w2 + w3 + w4;
        if (d < 0.001f)
            return;

        float pic_vel =
            (w1 * grid_main[(cx, cy)] +
            w2 * grid_main[(cx + 1, cy)] +
            w3 * grid_main[(cx + 1, cy + 1)] +
            w4 * grid_main[(cx + 1, cy)]) / d;

        float corr =
            (w1 * (grid_main[(cx, cy)] - grid_old[(cx, cy)]) +
            w2 * (grid_main[(cx + 1, cy)] - grid_old[(cx + 1, cy)]) +
            w3 * (grid_main[(cx + 1, cy + 1)] - grid_old[(cx + 1, cy + 1)]) +
            w4 * (grid_main[(cx, cy + 1)] - grid_old[(cx, cy + 1)])) / d;

        if (transfer_vx == 1)
            point.vx = (1.0f - FLIP_RATIO) * pic_vel + FLIP_RATIO * (point.vx + corr);
        else
            point.vy = (1.0f - FLIP_RATIO) * pic_vel + FLIP_RATIO * (point.vy + corr);

        points[ind] = point;
    }
    #endregion

    private void Init()
    {
        Random rand = new();
        Point[] temp_points = new Point[POINTS_COUNT];
        /*
        for (int i = 0; i < POINTS_COUNT ; i++)
        {
            Point point = new Point
            {
                x = 0.4f * WIDTH + rand.NextSingle() * 0.3f * WIDTH,
                y = rand.NextSingle() * HEIGHT,
            };
            temp_points[i] = point;
        }*/

        for (int i = 0; i < POINTS_COUNT; i++)
        {
            Point point = new Point
            {
                x = i > POINTS_COUNT / 2 ? (0.7f * WIDTH + rand.NextSingle() * 0.3f * WIDTH) : (rand.NextSingle() * 0.3f * WIDTH),
                y = rand.NextSingle() * HEIGHT,
            };
            temp_points[i] = point;
        }

        main_points.CopyFromCPU(temp_points);

        byte a = (byte)((POINTS_COUNT >> 24) & 255);
        byte b = (byte)((POINTS_COUNT >> 16) & 255);
        byte c = (byte)((POINTS_COUNT >> 8) & 255);
        byte d = (byte)((POINTS_COUNT) & 255);

        int offset = CHUNKS_X * CHUNKS_Y;
        buckets_total.MemSet(accelerator.DefaultStream, a, offset, 4);
        buckets_total.MemSet(accelerator.DefaultStream, b, offset, 3);
        buckets_total.MemSet(accelerator.DefaultStream, c, offset, 2);
        buckets_total.MemSet(accelerator.DefaultStream, d, offset, 1);
    }

    public void Dispose()
    {
        grid_delta_vy.Dispose();
        grid_old_vy.Dispose();
        grid_main_vy.Dispose();

        grid_delta_vx.Dispose();
        grid_old_vx.Dispose();
        grid_main_vx.Dispose();

        fluid_cells_total.Dispose();
        fluid_cells.Dispose();

        densities_total.Dispose();
        densities.Dispose();

        buckets_total.Dispose();
        buckets_used.Dispose();

        swap_points.Dispose();
        main_points.Dispose();

        accelerator.Dispose();
        context.Dispose();
    }

    public struct Point
    {
        public float x, y, vx, vy;
    }
}
