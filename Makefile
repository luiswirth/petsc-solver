default: solve

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

solve: solve.o
	-${CLINKER} -o solve solve.o ${SLEPC_EPS_LIB}
	${RM} solve.o
