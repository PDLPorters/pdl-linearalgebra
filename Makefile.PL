use ExtUtils::MakeMaker;
use Config;
use Devel::CheckLib qw(check_lib_or_exit);
use ExtUtils::F77;

our %ldloadlibs = ($^O =~ /MSWin/ && $Config{cc} eq 'cl')
  ? (LDLOADLIBS => 'oldnames.lib kernel32.lib user32.lib gdi32.lib winspool.lib  comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib  netapi32.lib uuid.lib wsock32.lib mpr.lib winmm.lib  version.lib odbc32.lib odbccp32.lib msvcrt.lib  ../lapack/libacml.lib "C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\lib\msvcrt.lib" ')
  : ();
my @pkgs = qw(lapack blas);
our $libs0 = (
  eval {require PkgConfig; join ' ', map PkgConfig->find($_)->get_ldflags, @pkgs} ||
  eval {require ExtUtils::PkgConfig; join ' ', map ExtUtils::PkgConfig->libs($_), @pkgs} ||
  `pkg-config @pkgs --libs` ||
  '-L/usr/lib/atlas -llapack -lblas -latlas'
) . ' ' . ExtUtils::F77->runtime;
{
  # work around Devel::CheckLib not doing Text::ParseWords::shellwords
  use Text::ParseWords qw(shellwords);
  my @libpath = map {my $r=$_;$r=~s/^-L//?$r:()} shellwords $libs0;
  check_lib_or_exit(
    lib => 'lapack',
    libpath => \@libpath,
    header => [($^O =~ /MSWin/ ? 'float.h' : ()), qw(stdio.h math.h)],
    function => <<'EOF',
long c_zero = 0;
long c_nine = 9;
extern long ilaenv_(long *ispec, char *name__, char *opts, long *n1,
  long *n2, long *n3, long *n4, long name_len, long opts_len);
long i = ilaenv_(&c_nine, "SGESDD", " ", &c_zero, &c_zero, &c_zero, &c_zero, 6, 1);
if (argc > 2) printf("%ld", i); /* try to stop optimiser eliminating */
EOF
  );
}

WriteMakefile(
  NAME => 'PDL::LinearAlgebra',
  ABSTRACT => 'PDL bindings to some BLAS and LAPACK library routines',
  AUTHOR => [ 'Chris Marshall <chm@cpan.org>' ],
  VERSION_FROM => 'lib/PDL/LinearAlgebra.pm',
  LICENSE => 'artistic_2',
  META_MERGE => {
    "meta-spec" => { version => 2 },
    resources => {
      homepage => 'http://pdl.perl.org/',
      repository  => {
        url => 'git://github.com/PDLPorters/pdl-linearalgebra.git',
        type => 'git',
        web => 'https://github.com/PDLPorters/pdl-linearalgebra',
      },
      bugtracker => {web=>'https://github.com/PDLPorters/pdl-linearalgebra/issues'},
    },
    x_IRC => 'irc://irc.perl.org/#pdl',
  },
  CONFIGURE_REQUIRES =>  {
    "PDL" => '2.075',
    "Devel::CheckLib" => 0,
    "ExtUtils::F77" => 0,
  },
  PREREQ_PM => {
    "PDL" => '2.075', # RedoDimsCode with SIZE() available
  },
  TEST_REQUIRES =>  {
    "Test::More" => '0.88', # done_testing
  },
  dist => { PREOP=>'$(PERL) -MPDL::Core::Dev -e pdlpp_mkgen $(DISTVNAME)' }, # GENERATED subdir in dist tarball
  clean => { FILES => '*~' },
);
