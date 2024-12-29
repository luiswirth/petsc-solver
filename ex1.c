#include <slepceps.h>

static char help[] = "My example\n";

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  PetscInt n=30;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL); CHKERRQ(ierr);

  Mat A;
  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  PetscInt Istart,Iend;
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);
  for (int i = Istart; i < Iend; ++i) {
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

  Vec xr, xi;
  ierr = MatCreateVecs(A,NULL,&xr); CHKERRQ(ierr);
  ierr = MatCreateVecs(A,NULL,&xi); CHKERRQ(ierr);

  EPS eps;
  ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, A, NULL); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);

  PetscInt its;
  ierr = EPSGetIterationNumber(eps,&its); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"niterations=%d\n",its); CHKERRQ(ierr);

  PetscInt nconv;
  ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"neigenpairs=%d\n",nconv); CHKERRQ(ierr);

  PetscScalar kr, ki;
  PetscReal error,re,im;
  for (int i = 0; i < nconv; ++i) {
    ierr = EPSGetEigenpair(eps, i, &kr, &ki, xr, xi); CHKERRQ(ierr);

    ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error); CHKERRQ(ierr);

    #if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
    #else
      re = kr;
      im = ki;
    #endif

    if (im != 0.0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%9f%+9fi %12g\n",(double)re,(double)im,(double)error); CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%12f %12g\n",(double)re,(double)error); CHKERRQ(ierr);
    }
    
  }

  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = VecDestroy(&xr); CHKERRQ(ierr);
  ierr = VecDestroy(&xi); CHKERRQ(ierr);

  ierr = SlepcFinalize();
  return ierr;
}
