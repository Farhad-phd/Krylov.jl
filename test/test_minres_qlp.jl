@testset "minres_qlp" begin
  minres_qlp_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Cubic spline matrix.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = minres_qlp(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(stats.status == "on trust-region boundary")
        @test(abs(norm(x) - radius) ≤ minres_qlp_tol * radius)
      end

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      if FC == Float64
        radius = 0.75 * norm(x)
        (x, stats) = minres_qlp(A, b, radius=radius, itmax=10)
        @test(stats.solved)
        @test(stats.status == "on trust-region boundary")
        @test(abs(norm(x) - radius) ≤ minres_qlp_tol * radius)
      end

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = minres_qlp(A, b)
      @test norm(x) == 0
      @test stats.status == "x is a zero-residual solution"

      # Singular inconsistent system
      A, b = square_inconsistent(FC=FC)
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Symmetric inconsistent system
      A, b = symmetric_inconsistent()
      (x, stats) = minres_qlp(A, b)
      r = b - A * x
      Aresid = norm(A*r) / norm(A*b)
      @test(Aresid ≤ minres_qlp_tol)
      @test stats.inconsistent

      # Shifted system
      A, b = symmetric_indefinite(FC=FC)
      λ = 2.0
      (x, stats) = minres_qlp(A, b, λ=λ)
      r = b - (A + λ*I) * x
      resid = norm(r) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = minres_qlp(A, b, M=M)
      r = b - A * x
      resid = sqrt(real(dot(r, M * r))) / norm(b)
      @test(resid ≤ minres_qlp_tol * norm(A) * norm(x))
      @test(stats.solved)

      # A large radius exercises multiple preconditioned iterations without
      # allowing the trust-region scratch vector to alter the Lanczos vectors.
      M = Diagonal(diag(M))
      x_ref = copy(x)
      x_ref_norm = sqrt(real(dot(x_ref, M \ x_ref)))
      radius = 100 * max(x_ref_norm, real(one(FC)))
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b; M=M, radius=radius)
      @test workspace.stats.solved
      @test workspace.stats.status != "on trust-region boundary"
      @test length(workspace.ztmp) == length(b)
      @test norm(workspace.x - x_ref) ≤ minres_qlp_tol * max(norm(x_ref), real(one(FC)))

      radius = x_ref_norm / 2
      minres_qlp!(workspace, A, b; M=M, radius=radius)
      x_norm = sqrt(real(dot(workspace.x, M \ workspace.x)))
      @test workspace.stats.status == "on trust-region boundary"
      @test abs(x_norm - radius) ≤ minres_qlp_tol * radius

      # test callback function
      A, b = sparse_laplacian(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      tol = 1.0
      cb_n2 = TestCallbackN2(A, b, tol = tol)
      minres_qlp!(workspace, A, b, atol = 0.0, rtol = 0.0, Artol = 0.0, callback = cb_n2)
      @test workspace.stats.status == "user-requested exit"
      @test cb_n2(workspace)

      # Test linesearch
      A, b = symmetric_indefinite(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.indefinite == true
      # Verify that the returned direction indeed exhibits nonpositive curvature.
      # For both real and complex cases, ensure to take the real part.
      @test real(dot(npc_dir, A * npc_dir)) <= 0
    
      # Test Linesearch which would stop on the first call since A is negative definite
      A, b = symmetric_indefinite(FC=FC; shift = 5)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test stats.niter == 1 
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test stats.npcCount == 1
      @test real(dot(npc_dir, A * npc_dir)) <= 0      

      # Test when b^TAb=0 and linesearch is true
      A, b = system_zero_quad(FC=FC)
      workspace = MinresQlpWorkspace(A, b)
      minres_qlp!(workspace, A, b, linesearch=true)
      x, stats, npc_dir = workspace.x, workspace.stats, workspace.npc_dir
      @test stats.status == "nonpositive curvature"
      @test all(x .== b)
      @test stats.solved == true
      @test stats.indefinite == true
      @test real(dot(npc_dir, A * npc_dir)) ≈ 0.0

      # Test if warm_start and linesearch are both true, it should throw an error
      A, b = symmetric_indefinite(FC=FC)
      @test_throws MethodError minres_qlp(A, b, warm_start = true, linesearch = true)

      # Test radius > 0 and b^TAb = 0
      A, b = zero_rhs(FC=FC)
      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b, radius = 10 * real(one(FC)))
      x, stats = solver.x, solver.stats
      @test stats.status == "x is a zero-residual solution"
      @test norm(x) == zero(FC)
      @test stats.niter == 0

      # Test radius > 0 and nonpositive curvature at the first iteration
      A, b = symmetric_indefinite(FC=FC; shift=5)
      radius = real(0.5 * norm(b))
      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b; radius=radius)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      @test stats.niter == 1
      @test stats.npcCount == 1
      @test stats.indefinite == true
      @test stats.status == "on trust-region boundary"
      @test abs(norm(x) - radius) ≤ minres_qlp_tol * radius
      @test norm(x - radius / norm(npc_dir) * npc_dir) ≤ minres_qlp_tol * radius

      # Test radius > 0 and negative curvature detected
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      linesearch_solver = MinresQlpWorkspace(A, b)
      minres_qlp!(linesearch_solver, A, b; linesearch=true)
      x_previous = copy(linesearch_solver.x)
      @test linesearch_solver.stats.niter > 1

      previous_solver = MinresQlpWorkspace(A, b)
      minres_qlp!(previous_solver, A, b; atol=zero(real(FC)), rtol=zero(real(FC)),
          Artol=zero(real(FC)), itmax=linesearch_solver.stats.niter - 1)
      @test previous_solver.stats.niter == linesearch_solver.stats.niter - 1
      @test norm(x_previous - previous_solver.x) ≤ minres_qlp_tol * max(norm(x_previous), real(one(FC)))

      solver = MinresQlpWorkspace(A, b)
      radius = 10 * real(one(FC))
      minres_qlp!(solver, A, b; radius=radius, history=true)
      x, stats, npc_dir = solver.x, solver.stats, solver.npc_dir
      step = x - x_previous
      t = real(dot(npc_dir, step)) / real(dot(npc_dir, npc_dir))
      @test stats.niter == linesearch_solver.stats.niter > 1
      @test stats.npcCount == 2
      @test stats.status == "on trust-region boundary"
      @test stats.indefinite == true
      @test length(stats.residuals) == stats.niter + 1
      @test abs(norm(x) - radius) ≤ minres_qlp_tol * radius
      @test t > 0
      @test norm(step - t * npc_dir) ≤ minres_qlp_tol * max(norm(step), real(one(FC)))
      @test real(dot(npc_dir, A * npc_dir)) ≤ minres_qlp_tol * norm(npc_dir)^2

      x_minres, stats_minres = minres(A, b; radius=radius)
      @test stats_minres.status == "on trust-region boundary"
      @test norm(x - x_minres) ≤ minres_qlp_tol * radius

      # Test that minres_qlp throws an error when radius > 0 and linesearch is true
      A, b = symmetric_indefinite(FC = FC, shift = 5)
      @test_throws ErrorException minres_qlp(A, b, radius = real(one(FC)), linesearch = true)

      @test_throws TypeError minres_qlp(A, b, callback = workspace -> "string", history = true)

      # Test: Ensure stats are reset when reusing workspace
      # (pᵀAp < 0)
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]
      
      # Initialize workspace and solve
      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b; linesearch=true)
      
      # Verify the "npc" state was recorded
      @test solver.stats.npcCount == 1
      @test solver.stats.indefinite == true
      @test solver.stats.status == "nonpositive curvature"

      # Reuse the SAME solver on a Positive Definite System

      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 1.0
      ]
      b = FC[1.0, 1.0, 1.0, 1.0]

      # Run the solver again on the same workspace
      minres_qlp!(solver, A, b; linesearch=true)

      # Verify the RESET works
      @test solver.stats.npcCount == 0
      @test solver.stats.indefinite == false
      @test solver.stats.solved == true

      # Test: Ensure stats are reset when reusing workspace with radius > 0
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 -1.0
      ]
      b = FC[1.0, 1.0, 1.0, 0.1]

      solver = MinresQlpWorkspace(A, b)
      minres_qlp!(solver, A, b; radius = 10 * real(one(FC)))
      @test solver.stats.npcCount == 2
      @test solver.stats.indefinite == true
      @test solver.stats.status == "on trust-region boundary"
      @test abs(norm(solver.x) - 10) ≤ 10 * minres_qlp_tol

      # Reuse the SAME solver on a Positive Definite System with radius > 0
      A = FC[
        10.0 0.0 0.0 0.0;
        0.0 8.0 0.0 0.0;
        0.0 0.0 5.0 0.0;
        0.0 0.0 0.0 1.0
      ]
      b = FC[1.0, 1.0, 1.0, 1.0]

      # Small radius forces the iterate to the boundary.
      minres_qlp!(solver, A, b; radius = real(0.1 * one(FC)))
      @test solver.stats.npcCount == 0
      @test solver.stats.indefinite == false
      @test solver.stats.status == "on trust-region boundary"
      @test abs(norm(solver.x) - 0.1) ≤ minres_qlp_tol
    end
  end
end
