function test_krylov_workspaces(FC; krylov_constructor::Bool=false, use_val::Bool=true)
  A   = FC.(get_div_grad(4, 4, 4))  # Dimension n x n
  n   = size(A, 1)
  m   = div(n, 2)
  Au  = A[1:m,:]  # Dimension m x n
  Ao  = A[:,1:m]  # Dimension n x m
  b   = Ao * ones(FC, m) # Dimension n
  c   = Au * ones(FC, n) # Dimension m
  mem = 10
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  T = real(FC)
  S = Vector{FC}

  workspaces = Dict{Symbol, KrylovWorkspace}()
  if use_val
    if krylov_constructor
      kc_nn = KrylovConstructor(b)
      kc_mn = KrylovConstructor(c, b)
      kc_nm = KrylovConstructor(b, c)
      workspaces[:cg] = @inferred krylov_workspace(Val(:cg), kc_nn)
      workspaces[:car] = @inferred krylov_workspace(Val(:car), kc_nn)
      workspaces[:symmlq] = @inferred krylov_workspace(Val(:symmlq), kc_nn)
      workspaces[:minres] = @inferred krylov_workspace(Val(:minres), kc_nn)
      workspaces[:minares] = @inferred krylov_workspace(Val(:minares), kc_nn)
      workspaces[:cg_lanczos] = @inferred krylov_workspace(Val(:cg_lanczos), kc_nn)
      workspaces[:diom] = @inferred krylov_workspace(Val(:diom), kc_nn; memory=mem)
      workspaces[:fom] = @inferred krylov_workspace(Val(:fom), kc_nn; memory=mem)
      workspaces[:dqgmres] = @inferred krylov_workspace(Val(:dqgmres), kc_nn; memory=mem)
      workspaces[:gmres] = @inferred krylov_workspace(Val(:gmres), kc_nn; memory=mem)
      workspaces[:gpmr] = @inferred krylov_workspace(Val(:gpmr), kc_nm; memory=mem)
      workspaces[:fgmres] = @inferred krylov_workspace(Val(:fgmres), kc_nn; memory=mem)
      workspaces[:cr] = @inferred krylov_workspace(Val(:cr), kc_nn)
      workspaces[:crmr] = @inferred krylov_workspace(Val(:crmr), kc_mn)
      workspaces[:cgs] = @inferred krylov_workspace(Val(:cgs), kc_nn)
      workspaces[:bicgstab] = @inferred krylov_workspace(Val(:bicgstab), kc_nn)
      workspaces[:craigmr] = @inferred krylov_workspace(Val(:craigmr), kc_mn)
      workspaces[:cgne] = @inferred krylov_workspace(Val(:cgne), kc_mn)
      workspaces[:lnlq] = @inferred krylov_workspace(Val(:lnlq), kc_mn)
      workspaces[:craig] = @inferred krylov_workspace(Val(:craig), kc_mn)
      workspaces[:lslq] = @inferred krylov_workspace(Val(:lslq), kc_nm)
      workspaces[:cgls] = @inferred krylov_workspace(Val(:cgls), kc_nm)
      workspaces[:lsqr] = @inferred krylov_workspace(Val(:lsqr), kc_nm)
      workspaces[:crls] = @inferred krylov_workspace(Val(:crls), kc_nm)
      workspaces[:lsmr] = @inferred krylov_workspace(Val(:lsmr), kc_nm)
      workspaces[:usymqr] = @inferred krylov_workspace(Val(:usymqr), kc_nm)
      workspaces[:trilqr] = @inferred krylov_workspace(Val(:trilqr), kc_nn)
      workspaces[:bilq] = @inferred krylov_workspace(Val(:bilq), kc_nn)
      workspaces[:bilqr] = @inferred krylov_workspace(Val(:bilqr), kc_nn)
      workspaces[:minres_qlp] = @inferred krylov_workspace(Val(:minres_qlp), kc_nn)
      workspaces[:qmr] = @inferred krylov_workspace(Val(:qmr), kc_nn)
      workspaces[:usymlq] = @inferred krylov_workspace(Val(:usymlq), kc_mn)
      workspaces[:tricg] = @inferred krylov_workspace(Val(:tricg), kc_mn)
      workspaces[:trimr] = @inferred krylov_workspace(Val(:trimr), kc_mn)
      workspaces[:cg_lanczos_shift] = @inferred krylov_workspace(Val(:cg_lanczos_shift), kc_nn, nshifts)
      workspaces[:cgls_lanczos_shift] = @inferred krylov_workspace(Val(:cgls_lanczos_shift), kc_nm, nshifts)
    else
      workspaces[:cg] = @inferred krylov_workspace(Val(:cg), n, n, S)
      workspaces[:car] = @inferred krylov_workspace(Val(:car), n, n, S)
      workspaces[:symmlq] = @inferred krylov_workspace(Val(:symmlq), n, n, S)
      workspaces[:minres] = @inferred krylov_workspace(Val(:minres), n, n, S)
      workspaces[:minares] = @inferred krylov_workspace(Val(:minares), n, n, S)
      workspaces[:cg_lanczos] = @inferred krylov_workspace(Val(:cg_lanczos), n, n, S)
      workspaces[:diom] = @inferred krylov_workspace(Val(:diom), n, n, S; memory=mem)
      workspaces[:fom] = @inferred krylov_workspace(Val(:fom), n, n, S; memory=mem)
      workspaces[:dqgmres] = @inferred krylov_workspace(Val(:dqgmres), n, n, S; memory=mem)
      workspaces[:gmres] = @inferred krylov_workspace(Val(:gmres), n, n, S; memory=mem)
      workspaces[:gpmr] = @inferred krylov_workspace(Val(:gpmr), n, m, S; memory=mem)
      workspaces[:fgmres] = @inferred krylov_workspace(Val(:fgmres), n, n, S; memory=mem)
      workspaces[:cr] = @inferred krylov_workspace(Val(:cr), n, n, S)
      workspaces[:crmr] = @inferred krylov_workspace(Val(:crmr), m, n, S)
      workspaces[:cgs] = @inferred krylov_workspace(Val(:cgs), n, n, S)
      workspaces[:bicgstab] = @inferred krylov_workspace(Val(:bicgstab), n, n, S)
      workspaces[:craigmr] = @inferred krylov_workspace(Val(:craigmr), m, n, S)
      workspaces[:cgne] = @inferred krylov_workspace(Val(:cgne), m, n, S)
      workspaces[:lnlq] = @inferred krylov_workspace(Val(:lnlq), m, n, S)
      workspaces[:craig] = @inferred krylov_workspace(Val(:craig), m, n, S)
      workspaces[:lslq] = @inferred krylov_workspace(Val(:lslq), n, m, S)
      workspaces[:cgls] = @inferred krylov_workspace(Val(:cgls), n, m, S)
      workspaces[:lsqr] = @inferred krylov_workspace(Val(:lsqr), n, m, S)
      workspaces[:crls] = @inferred krylov_workspace(Val(:crls), n, m, S)
      workspaces[:lsmr] = @inferred krylov_workspace(Val(:lsmr), n, m, S)
      workspaces[:usymqr] = @inferred krylov_workspace(Val(:usymqr), n, m, S)
      workspaces[:trilqr] = @inferred krylov_workspace(Val(:trilqr), n, n, S)
      workspaces[:bilq] = @inferred krylov_workspace(Val(:bilq), n, n, S)
      workspaces[:bilqr] = @inferred krylov_workspace(Val(:bilqr), n, n, S)
      workspaces[:minres_qlp] = @inferred krylov_workspace(Val(:minres_qlp), n, n, S)
      workspaces[:qmr] = @inferred krylov_workspace(Val(:qmr), n, n, S)
      workspaces[:usymlq] = @inferred krylov_workspace(Val(:usymlq), m, n, S)
      workspaces[:tricg] = @inferred krylov_workspace(Val(:tricg), m, n, S)
      workspaces[:trimr] = @inferred krylov_workspace(Val(:trimr), m, n, S)
      workspaces[:cg_lanczos_shift] = @inferred krylov_workspace(Val(:cg_lanczos_shift), n, n, nshifts, S)
      workspaces[:cgls_lanczos_shift] = @inferred krylov_workspace(Val(:cgls_lanczos_shift), n, m, nshifts, S)
    end
  else
    if krylov_constructor
      kc_nn = KrylovConstructor(b)
      kc_mn = KrylovConstructor(c, b)
      kc_nm = KrylovConstructor(b, c)
      workspaces[:cg] = krylov_workspace(:cg, kc_nn)
      workspaces[:car] = krylov_workspace(:car, kc_nn)
      workspaces[:symmlq] = krylov_workspace(:symmlq, kc_nn)
      workspaces[:minres] = krylov_workspace(:minres, kc_nn)
      workspaces[:minares] = krylov_workspace(:minares, kc_nn)
      workspaces[:cg_lanczos] = krylov_workspace(:cg_lanczos, kc_nn)
      workspaces[:diom] = krylov_workspace(:diom, kc_nn; memory=mem)
      workspaces[:fom] = krylov_workspace(:fom, kc_nn; memory=mem)
      workspaces[:dqgmres] = krylov_workspace(:dqgmres, kc_nn; memory=mem)
      workspaces[:gmres] = krylov_workspace(:gmres, kc_nn; memory=mem)
      workspaces[:gpmr] = krylov_workspace(:gpmr, kc_nm; memory=mem)
      workspaces[:fgmres] = krylov_workspace(:fgmres, kc_nn; memory=mem)
      workspaces[:cr] = krylov_workspace(:cr, kc_nn)
      workspaces[:crmr] = krylov_workspace(:crmr, kc_mn)
      workspaces[:cgs] = krylov_workspace(:cgs, kc_nn)
      workspaces[:bicgstab] = krylov_workspace(:bicgstab, kc_nn)
      workspaces[:craigmr] = krylov_workspace(:craigmr, kc_mn)
      workspaces[:cgne] = krylov_workspace(:cgne, kc_mn)
      workspaces[:lnlq] = krylov_workspace(:lnlq, kc_mn)
      workspaces[:craig] = krylov_workspace(:craig, kc_mn)
      workspaces[:lslq] = krylov_workspace(:lslq, kc_nm)
      workspaces[:cgls] = krylov_workspace(:cgls, kc_nm)
      workspaces[:lsqr] = krylov_workspace(:lsqr, kc_nm)
      workspaces[:crls] = krylov_workspace(:crls, kc_nm)
      workspaces[:lsmr] = krylov_workspace(:lsmr, kc_nm)
      workspaces[:usymqr] = krylov_workspace(:usymqr, kc_nm)
      workspaces[:trilqr] = krylov_workspace(:trilqr, kc_nn)
      workspaces[:bilq] = krylov_workspace(:bilq, kc_nn)
      workspaces[:bilqr] = krylov_workspace(:bilqr, kc_nn)
      workspaces[:minres_qlp] = krylov_workspace(:minres_qlp, kc_nn)
      workspaces[:qmr] = krylov_workspace(:qmr, kc_nn)
      workspaces[:usymlq] = krylov_workspace(:usymlq, kc_mn)
      workspaces[:tricg] = krylov_workspace(:tricg, kc_mn)
      workspaces[:trimr] = krylov_workspace(:trimr, kc_mn)
      workspaces[:cg_lanczos_shift] = krylov_workspace(:cg_lanczos_shift, kc_nn, nshifts)
      workspaces[:cgls_lanczos_shift] = krylov_workspace(:cgls_lanczos_shift, kc_nm, nshifts)
    else
      workspaces[:cg] = krylov_workspace(:cg, n, n, S)
      workspaces[:car] = krylov_workspace(:car, n, n, S)
      workspaces[:symmlq] = krylov_workspace(:symmlq, n, n, S)
      workspaces[:minres] = krylov_workspace(:minres, n, n, S)
      workspaces[:minares] = krylov_workspace(:minares, n, n, S)
      workspaces[:cg_lanczos] = krylov_workspace(:cg_lanczos, n, n, S)
      workspaces[:diom] = krylov_workspace(:diom, n, n, S; memory=mem)
      workspaces[:fom] = krylov_workspace(:fom, n, n, S; memory=mem)
      workspaces[:dqgmres] = krylov_workspace(:dqgmres, n, n, S; memory=mem)
      workspaces[:gmres] = krylov_workspace(:gmres, n, n, S; memory=mem)
      workspaces[:gpmr] = krylov_workspace(:gpmr, n, m, S; memory=mem)
      workspaces[:fgmres] = krylov_workspace(:fgmres, n, n, S; memory=mem)
      workspaces[:cr] = krylov_workspace(:cr, n, n, S)
      workspaces[:crmr] = krylov_workspace(:crmr, m, n, S)
      workspaces[:cgs] = krylov_workspace(:cgs, n, n, S)
      workspaces[:bicgstab] = krylov_workspace(:bicgstab, n, n, S)
      workspaces[:craigmr] = krylov_workspace(:craigmr, m, n, S)
      workspaces[:cgne] = krylov_workspace(:cgne, m, n, S)
      workspaces[:lnlq] = krylov_workspace(:lnlq, m, n, S)
      workspaces[:craig] = krylov_workspace(:craig, m, n, S)
      workspaces[:lslq] = krylov_workspace(:lslq, n, m, S)
      workspaces[:cgls] = krylov_workspace(:cgls, n, m, S)
      workspaces[:lsqr] = krylov_workspace(:lsqr, n, m, S)
      workspaces[:crls] = krylov_workspace(:crls, n, m, S)
      workspaces[:lsmr] = krylov_workspace(:lsmr, n, m, S)
      workspaces[:usymqr] = krylov_workspace(:usymqr, n, m, S)
      workspaces[:trilqr] = krylov_workspace(:trilqr, n, n, S)
      workspaces[:bilq] = krylov_workspace(:bilq, n, n, S)
      workspaces[:bilqr] = krylov_workspace(:bilqr, n, n, S)
      workspaces[:minres_qlp] = krylov_workspace(:minres_qlp, n, n, S)
      workspaces[:qmr] = krylov_workspace(:qmr, n, n, S)
      workspaces[:usymlq] = krylov_workspace(:usymlq, m, n, S)
      workspaces[:tricg] = krylov_workspace(:tricg, m, n, S)
      workspaces[:trimr] = krylov_workspace(:trimr, m, n, S)
      workspaces[:cg_lanczos_shift] = krylov_workspace(:cg_lanczos_shift, n, n, nshifts, S)
      workspaces[:cgls_lanczos_shift] = krylov_workspace(:cgls_lanczos_shift, n, m, nshifts, S)
    end
  end

  @testset "Check compatibility between each KrylovWorkspace and the dimension of the linear problem" begin
    A2  = FC.(get_div_grad(2, 2, 2))
    n2  = size(A2, 1)
    m2  = div(n2, 2)
    Au2 = A2[1:m2,:]
    Ao2 = A2[:,1:m2]
    b2  = Ao2 * ones(FC, m2)
    c2  = Au2 * ones(FC, n2)
    shifts2 = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
    for (method, workspace) in workspaces
      if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr)
        @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(workspace, A2, b2)
      end
      method == :cg_lanczos_shift && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(workspace, A2, b2, shifts2)
      method == :cg_lanczos_shift && @test_throws ErrorException("workspace.nshifts = $(workspace.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") krylov_solve!(workspace, A, b, shifts2)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m2, $n2)") krylov_solve!(workspace, Au2, c2)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(workspace, Ao2, b2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(workspace, Ao2, b2, shifts2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("workspace.nshifts = $(workspace.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") krylov_solve!(workspace, Ao, b, shifts2)
      method ∈ (:bilqr, :trilqr) && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $n2)") krylov_solve!(workspace, A2, b2, b2)
      method == :gpmr && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(workspace, Ao2, Au2, b2, c2)
      method ∈ (:tricg, :trimr) && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(workspace, Ao2, b2, c2)
      method == :usymlq && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m2, $n2)") krylov_solve!(workspace, Au2, c2, b2)
      method == :usymqr && @test_throws ErrorException("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($n2, $m2)") krylov_solve!(workspace, Ao2, b2, c2)
    end
  end

  @testset "Test the keyword argument timemax" begin
    timemax = 0.0
    for (method, workspace) in workspaces
      method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr) && krylov_solve!(workspace, A, b, timemax=timemax)
      method == :cg_lanczos_shift && krylov_solve!(workspace, A, b, shifts, timemax=timemax)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && krylov_solve!(workspace, Au, c, timemax=timemax)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && krylov_solve!(workspace, Ao, b, timemax=timemax)
      method == :cgls_lanczos_shift && krylov_solve!(workspace, Ao, b, shifts, timemax=timemax)
      method ∈ (:bilqr, :trilqr) && krylov_solve!(workspace, A, b, b, timemax=timemax)
      method == :gpmr && krylov_solve!(workspace, Ao, Au, b, c, timemax=timemax)
      method ∈ (:tricg, :trimr) && krylov_solve!(workspace, Au, c, b, timemax=timemax)
      method == :usymlq && krylov_solve!(workspace, Au, c, b, timemax=timemax)
      method == :usymqr && krylov_solve!(workspace, Ao, b, c, timemax=timemax)
      @test workspace.stats.status == "time limit exceeded"
    end
  end

  for (method, workspace) in workspaces
    @testset "$(method)" begin
      for i = 1 : 3
        A  = i * A
        Au = i * Au
        Ao = i * Ao
        b  = 5 * b
        c  = 3 * c

        if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom,
                     :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr, :cg_lanczos_shift)
          if method == :cg_lanczos_shift
            use_val ? @inferred(krylov_solve(Val(method), A, b, shifts)) : krylov_solve(method, A, b, shifts)
            @inferred krylov_solve!(workspace, A, b, shifts)
          else
            use_val ? @inferred(krylov_solve(Val(method), A, b)) : krylov_solve(method, A, b)
            @inferred krylov_solve!(workspace, A, b)
          end
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == (method ∈ (:cgs, :bicgstab) ? 2 * niter : niter)
          @test Atprod_count(workspace) == (method ∈ (:bilq, :qmr) ? niter : 0)
          @test solution(workspace) === workspace.x
          @test solution_count(workspace) == 1
        end

        if method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr)
          use_val ? @inferred(krylov_solve(Val(method), Au, c)) : krylov_solve(method, Au, c)
          @inferred krylov_solve!(workspace, Au, c)
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == niter
          @test Atprod_count(workspace) == niter
          @test solution(workspace, 1) === workspace.x
          @test solution_count(workspace) == (method ∈ (:cgne, :crmr) ? 1 : 2)
          (solution_count(workspace) == 2) && (@test solution(workspace, 2) == workspace.y)
        end

        if method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr, :cgls_lanczos_shift)
          if method == :cgls_lanczos_shift
            use_val ? @inferred(krylov_solve(Val(method), Ao, b, shifts)) : krylov_solve(method, Ao, b, shifts)
            @inferred krylov_solve!(workspace, Ao, b, shifts)
          else
            use_val ? @inferred(krylov_solve(Val(method), Ao, b)) : krylov_solve(method, Ao, b)
            @inferred krylov_solve!(workspace, Ao, b)
          end
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == niter
          @test Atprod_count(workspace) == niter
          @test solution(workspace) === workspace.x
          @test solution_count(workspace) == 1
        end

        if method ∈ (:bilqr, :trilqr)
          use_val ? @inferred(krylov_solve(Val(method), A, b, b)) : krylov_solve(method, A, b, b)
          @inferred krylov_solve!(workspace, A, b, b)
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == niter
          @test Atprod_count(workspace) == niter
          @test solution(workspace, 1) === workspace.x
          @test solution(workspace, 2) === workspace.y
          @test solution_count(workspace) == 2
          @test issolved_primal(workspace)
          @test issolved_dual(workspace)
        end

        if method ∈ (:tricg, :trimr, :gpmr)
          if method == :gpmr
            use_val ? @inferred(krylov_solve(Val(method), Ao, Au, b, c)) : krylov_solve(method, Ao, Au, b, c)
            @inferred krylov_solve!(workspace, Ao, Au, b, c)
          else
            use_val ? @inferred(krylov_solve(Val(method), Au, c, b)) : krylov_solve(method, Au, c, b)
            @inferred krylov_solve!(workspace, Au, c, b)
          end
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == niter
          method != :gpmr && (@test Atprod_count(workspace) == niter)
          method == :gpmr && (@test Atprod_count(workspace) == 0)
          @test solution(workspace, 1) === workspace.x
          @test solution(workspace, 2) === workspace.y
          @test solution_count(workspace) == 2
        end

        if method ∈ (:usymlq, :usymqr)
          if method == :usymlq
            use_val ? @inferred(krylov_solve(Val(method), Au, c, b)) : krylov_solve(method, Au, c, b)
            @inferred krylov_solve!(workspace, Au, c, b)
          else
            use_val ? @inferred(krylov_solve(Val(method), Ao, b, c)) : krylov_solve(method, Ao, b, c)
            @inferred krylov_solve!(workspace, Ao, b, c)
          end
          niter = iteration_count(workspace)
          @test Aprod_count(workspace) == niter
          @test Atprod_count(workspace) == niter
          @test solution(workspace) === workspace.x
          @test solution_count(workspace) == 1
        end

        @test niter > 0
        @test statistics(workspace) === workspace.stats
        @test issolved(workspace)
        @test elapsed_time(workspace) > 0
      end

      test_show(workspace)
    end
  end
end

function test_block_krylov_workspaces(FC; use_val::Bool=true)
  A = FC.(get_div_grad(4, 4, 4))  # Dimension n x n
  m, n = size(A)
  p = 8
  B = rand(FC, m, p)
  mem = 4
  T = real(FC)
  SV = Vector{FC}
  SM = Matrix{FC}

  workspaces = Dict{Symbol, BlockKrylovWorkspace}()
  if use_val
    workspaces[:block_minres] = @inferred krylov_workspace(Val(:block_minres), m, n, p, SV, SM)
    workspaces[:block_gmres] = @inferred krylov_workspace(Val(:block_gmres), m, n, p, SV, SM; memory=mem)
  else
    workspaces[:block_minres] = krylov_workspace(:block_minres, m, n, p, SV, SM)
    workspaces[:block_gmres] = krylov_workspace(:block_gmres, m, n, p, SV, SM; memory=mem)
  end

  for (method, workspace) in workspaces
    @testset "$(method)" begin
      for i = 1 : 3
        B = 5 * B
        use_val ? @inferred(krylov_solve(Val(method), A, B)) : krylov_solve(method, A, B)
        @inferred krylov_solve!(workspace, A, B)
        niter = iteration_count(workspace)
        @test Aprod_count(workspace) == niter
        @test Atprod_count(workspace) == 0
        @test solution(workspace) === workspace.X
        @test solution_count(workspace) == 1
        @test niter > 0
        @test statistics(workspace) === workspace.stats
        @test issolved(workspace)
        @test elapsed_time(workspace) > 0
      end

      test_show(workspace)
    end
  end
end

function test_show(workspace)
  io = IOBuffer()
  show(io, workspace, show_stats=false)
  showed = String(take!(io))

  # Test that the lines have the same length
  str = split(showed, '\n', keepempty=false)
  len_row = length(str[1])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_row, &, str)

  # Test that the columns have the same length
  str2 = split(showed, ['│','┌','┬','┐','├','┼','┤','└','┴','┴','┘','\n'], keepempty=false)
  len_col1 = length(str2[1])
  len_col2 = length(str2[2])
  len_col3 = length(str2[3])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col1, &, str2[1:3:end-2])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col2, &, str2[2:3:end-1])
  @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col3, &, str2[3:3:end])

  # Code coverage
  show(io, workspace, show_stats=true)
end

@testset "Krylov solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      for krylov_constructor in (false, true)
        for use_val in (false, true)
          test_krylov_workspaces(FC; krylov_constructor, use_val)
          test_krylov_workspaces(FC; krylov_constructor, use_val)
        end
      end
    end
  end
end

@testset "Block Krylov solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      for use_val in (false, true)
        test_block_krylov_workspaces(FC; use_val)
      end
    end
  end
end
