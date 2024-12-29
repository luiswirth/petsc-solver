#include <petscmat.h>

static char help[] = "Generate and save stiffness (A) and mass (B) matrices in PETSc binary format.\n";

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

  PetscInt n = 30; // Number of grid points
  ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);

  Mat A, B;
  PetscViewer viewerA, viewerB;

  PetscInt Istart, Iend;

  // --- Create Stiffness Matrix (A) ---
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);
  for (PetscInt i = Istart; i < Iend; ++i) {
    if (i > 0) {
      ierr = MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i < n - 1) {
      ierr = MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatSetValue(A, i, i, 2.0, INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // --- Create Mass Matrix (B) ---
  ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
  ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(B); CHKERRQ(ierr);
  ierr = MatSetUp(B); CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B, &Istart, &Iend); CHKERRQ(ierr);
  for (PetscInt i = Istart; i < Iend; ++i) {
    ierr = MatSetValue(B, i, i, 2.0 / 3.0, INSERT_VALUES); CHKERRQ(ierr); // Diagonal term
    if (i > 0) {
      ierr = MatSetValue(B, i, i - 1, 1.0 / 6.0, INSERT_VALUES); CHKERRQ(ierr); // Off-diagonal term
    }
    if (i < n - 1) {
      ierr = MatSetValue(B, i, i + 1, 1.0 / 6.0, INSERT_VALUES); CHKERRQ(ierr); // Off-diagonal term
    }
  }
  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // --- Save Matrices in Binary Format ---
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "A.bin", FILE_MODE_WRITE, &viewerA); CHKERRQ(ierr);
  ierr = MatView(A, viewerA); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerA); CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "B.bin", FILE_MODE_WRITE, &viewerB); CHKERRQ(ierr);
  ierr = MatView(B, viewerB); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewerB); CHKERRQ(ierr);

  // --- Cleanup ---
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
