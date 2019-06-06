
###############################################################################
#
#                            GENANN Configuration
#
###############################################################################

AC_PREREQ([2.63])
AC_INIT(
    [GENANN],
    [1.36.0],
    [lv@codeplea.com],
    [genann],
    [https://github.com/codeplea/genann])

AC_COPYRIGHT([AC_PACKAGE_NAME AC_PACKAGE_VERSION])

###############################################################################
#
#                            Initial Setup
#
###############################################################################

AC_CONFIG_HEADERS([include/config.h:include/config.h.in])
AC_PROG_CC([gcc clang icc cl])
AC_LANG(C)

###############################################################################
#
#                            Check Libraries
#
###############################################################################

# Look for the math library in the system.
AC_CHECK_LIB(
    [m],
    [main],
    [AC_SUBST(
        [LIBMATH],
        ["-lm"])
        AC_DEFINE(
            [HAVE_LIBMATH],
            [1],
            [Define to 1 if math library found.])
    ],
    [AC_MSG_FAILURE(
        [Unable to find the math library.]
    )],
    [-lm]
)

# Look for gmp.
AC_ARG_WITH(
    [gmp],
    [AS_HELP_STRING(
        [--with-gmp],
        [Enable experimental support for the GNU Multiple Precision Library]
    )],
    [],
    [with_gmp=no]
)

LIBGMP=
AS_IF(
    [test "x$with_gmp" != xno],
    AC_CHECK_LIB(
        [gmp],
        [main],
        [AC_SUBST(
            [LIBGMP],
            ["-lgmp"])
        AC_DEFINE(
            [HAVE_GMP],
            [1],
            [Define to 1 if gmp found.]
        )],
        [AC_MSG_FAILURE(
            [--with-gmp was specified, but the required library could not be found.]
        )],
        [-lgmp]))

AC_ARG_WITH(
    [mpfr],
    [AS_HELP_STRING(
        [--with-mpfr],
        [Enable experimental support for the Multiple-Precision Floating-Point library]
    )],
    [],
    [with_mpfr=no]
)

LIBMPFR=
AS_IF(
    [test "x$with_mpfr" != xno],
    AC_CHECK_LIB(
        [mpfr],
        [main],
        [AC_SUBST(
            [LIBMPFR],
            ["-lmpfr"])
        AC_DEFINE(
            [HAVE_MPFR],
            [1],
            [Define to 1 if mpfr found.]
        )],
        [AC_MSG_FAILURE(
            [--with-mpfr was specified, but the required library could not be found.]
        )],
        [-lmpfr]))

###############################################################################
#
#                            Set Output Variables
#
###############################################################################

AC_ARG_VAR(ACTIVATION_FUNCTION, [Specify the behavior of the network. (SIGMOID, LINEAR, THRESHOLD)])

AC_SUBST(PROGRAM_NAME, [AC_PACKAGE_TARNAME])
AC_SUBST(PROGRAM_VERSION, [AC_PACKAGE_VERSION])

###############################################################################
#
#                       Generate Configuration Files
#
###############################################################################

AC_OUTPUT([Makefile])
