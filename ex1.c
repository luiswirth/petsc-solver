#include <slepceps.h>

static char help[] = "Generalized Hermitian Eigenvalue Problem (GHEP).\n";

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  PetscInt n = 30;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);

  Mat A;
  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  PetscInt Istart, Iend;
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (PetscInt i = Istart; i < Iend; ++i) {
    if (i > 0) {
      ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i < n-1) {
      ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  Mat B;
  ierr = MatCreate(PETSC_COMM_WORLD,&B); CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(B); CHKERRQ(ierr);
  ierr = MatSetUp(B); CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&Istart,&Iend); CHKERRQ(ierr);
  for (PetscInt i = Istart; i < Iend; ++i) {
    ierr = MatSetValue(B, i, i, 2.0/3.0, INSERT_VALUES); CHKERRQ(ierr);
    if (i > 0) {
      ierr = MatSetValue(B, i, i-1, 1.0/6.0, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i < n-1) {
      ierr = MatSetValue(B, i, i+1, 1.0/6.0, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  EPS eps;
  ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, A, B); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);

  PetscInt niterations;
  ierr = EPSGetIterationNumber(eps, &niterations); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "niterations=%d\n", niterations); CHKERRQ(ierr);

  PetscInt npairs;
  ierr = EPSGetConverged(eps, &npairs); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "npairs=%d\n", npairs); CHKERRQ(ierr);

  Vec xr, xi;
  ierr = MatCreateVecs(A, NULL, &xr); CHKERRQ(ierr);
  ierr = MatCreateVecs(A, NULL, &xi); CHKERRQ(ierr);

  PetscScalar kr, ki;
  PetscReal error, re, im;

  for (PetscInt i = 0; i < npairs; ++i) {
    ierr = EPSGetEigenpair(eps, i, &kr, &ki, xr, xi); CHKERRQ(ierr);
    ierr = EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error); CHKERRQ(ierr);

    #if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
    #else
      re = kr;
      im = ki;
    #endif

    if (im != 0.0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%9f%+9fi %12g\n", (double)re, (double)im, (double)error); CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%12f %12g\n", (double)re, (double)error); CHKERRQ(ierr);
    }
  }

  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);
  ierr = VecDestroy(&xr); CHKERRQ(ierr);
  ierr = VecDestroy(&xi); CHKERRQ(ierr);

  ierr = SlepcFinalize();
  return ierr;
}
