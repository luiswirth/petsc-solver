#include <slepceps.h>


static char help[] = "Generalized Hermitian Eigenvalue Problem (GHEP).\n";

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  char fileA[PETSC_MAX_PATH_LEN], fileB[PETSC_MAX_PATH_LEN];
  PetscBool setA = PETSC_FALSE, setB = PETSC_FALSE;

  ierr = PetscOptionsGetString(NULL, NULL, "-fileA", fileA, sizeof(fileA), &setA); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-fileB", fileB, sizeof(fileB), &setB); CHKERRQ(ierr);

  if (!setA || !setB) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide the `-fileA` and `-fileB` options.");
  }

  Mat A;
  PetscViewer viewerA;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileA, FILE_MODE_READ, &viewerA); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatLoad(A, viewerA); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerA); CHKERRQ(ierr);

  Mat B;
  PetscViewer viewerB;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileB, FILE_MODE_READ, &viewerB); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
  ierr = MatSetFromOptions(B); CHKERRQ(ierr);
  ierr = MatLoad(B, viewerB); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerB); CHKERRQ(ierr);  EPS eps;

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
