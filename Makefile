default: ghiep

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ghiep: ghiep.o
	-${CLINKER} -o ghiep ghiep.o ${SLEPC_EPS_LIB}
	${RM} ghiep.o

