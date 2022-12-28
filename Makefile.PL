use ExtUtils::MakeMaker;
use Config;
use Devel::CheckLib qw(check_lib check_lib_or_exit);
use ExtUtils::F77;
use Text::ParseWords qw(shellwords);
use String::ShellQuote qw(shell_quote);
use strict;

our %ldloadlibs = ($^O =~ /MSWin/ && $Config{cc} eq 'cl')
  ? (LDLOADLIBS => 'oldnames.lib kernel32.lib user32.lib gdi32.lib winspool.lib  comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib  netapi32.lib uuid.lib wsock32.lib mpr.lib winmm.lib  version.lib odbc32.lib odbccp32.lib msvcrt.lib  ../lapack/libacml.lib "C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\lib\msvcrt.lib" ')
  : ();

my $f77_uscore = (ExtUtils::F77->trail_ ? '_' : '');
my $fortran_checklib = <<EOF
/* The first line must not be a preprocessor macro or Devel::CheckLib will fail
   because Devel::Checklib places this code on one line: main() { <code> }  */
#define F77_USCORE $f77_uscore
#define CONCAT_(A, B) A ## B
#define CONCAT(A, B) CONCAT_(A, B)
#define FORTRAN(name) CONCAT(name, F77_USCORE)

long c_zero = 0;
long c_nine = 9;
extern long FORTRAN(ilaenv)(long *ispec, char *name__, char *opts, long *n1,
  long *n2, long *n3, long *n4, long name_len, long opts_len);
long i = FORTRAN(ilaenv)(&c_nine, "SGESDD", " ", &c_zero, &c_zero, &c_zero, &c_zero, 6, 1);
if (argc > 2) printf("%ld", i); /* try to stop optimiser eliminating */

extern void FORTRAN(dgebal)(char job, int n, double *a, int lda, int ilo, int ihi, double *scale, int info);
if (argc > 2) FORTRAN(dgebal)(0, 0, NULL, 0, 0, 0, NULL, 0);

extern void FORTRAN(dgesdd)(char job, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork, int *iwork, int info);
if (argc > 2) FORTRAN(dgesdd)(0, 0, 0, NULL, 0, NULL, NULL, 0, NULL, 0, NULL, 0, NULL, 0);

return 0;
EOF
;

our $libs0 = find_libs(ExtUtils::F77->runtime, $ENV{PDL_LA_LDFLAGS});

# Prevent makemaker from complaining about LIBS arguments that are
# neither -L nor -l.  We (ab)use LDFLAGS for this purpose:
$ldloadlibs{LDFLAGS} .= ' ' . shell_quote(grep { m!^/! } shellwords($libs0));
$libs0 = shell_quote(grep { !m!^/! } shellwords($libs0));

our $inc = "-DF77_USCORE=$f77_uscore";

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
    "PDL" => '2.078',
    "Devel::CheckLib" => '1.16', # argc is not supplied in 1.05, but 1.16 works.
    "ExtUtils::F77" => '1.26',
    "Text::ParseWords" => 0,
    "String::ShellQuote" => 0,
  },
  PREREQ_PM => {
    "PDL" => '2.078', # PDL::Complex method aliases
  },
  TEST_REQUIRES =>  {
    "Test::More" => '0.88', # done_testing
  },
  dist => { PREOP=>'$(PERL) -MPDL::Core::Dev -e pdlpp_mkgen $(DISTVNAME)' }, # GENERATED subdir in dist tarball
  clean => { FILES => '*~' },
);

sub pkgconfig
{
  my @pkgs = @_;
  my $ret =
    eval {require PkgConfig; join ' ', map PkgConfig->find($_)->get_ldflags, @pkgs} ||
    eval {require ExtUtils::PkgConfig; join ' ', map ExtUtils::PkgConfig->libs($_), @pkgs} ||
    sub { my $p = `pkg-config --libs @pkgs 2>/dev/null` ; chomp $p; return $p }->();

  print STDERR "pkg-config found libs for '@pkgs': $ret\n" if ($ENV{PDL_LA_DEBUG} && $ret);

  return $ret;
}

# Strip -l, return the libs as an array.  Takes an array or scalar:
sub _l
{
  return map { s/^-l//; $_ } grep { /^-l/ } map { shellwords($_) } @_;
}

# Strip -L, return the dirs as an array.  Takes an array or scalar:
sub _L
{
  return grep { -d $_ } map { s/^-L//; $_ } grep { /^-L/ } map { shellwords($_) } @_;
}

# Return any ldflags that are neither -L nor -l as an array.
# Takes an array or scalar:
sub _ld_other
{
  return grep { !/^-[lL]/ } map { shellwords($_) } @_;
}

# Take 3 array refs: 
#   - lib search paths
#   - lib names
#   - linker args that are neither -L nor -l
# Return a string of "-L<dirs> ... -Xfoo ... -l<lib> ..."
sub _merge_ld_flags
{
  my ($dirs, $libs, $ld_other) = @_;

  $dirs //= [];
  $libs //= [];
  $ld_other //= [];

  $dirs = shell_quote(map {"-L$_"} @$dirs);

  # If the libs start with a / then put them in ld_other, otherwise add -l in front:
  my @libfiles = grep { m!^/! } @$libs;
  my @libnames = grep { !m!^/! } @$libs;

  $libs = shell_quote(map { "-l$_" } @libnames);
  $ld_other = shell_quote(@$ld_other, @libfiles);

  # Join the ldflags only if they have content:
  my $ldflags = join (' ', grep { $_ } ($dirs, $ld_other, $libs));

  # RT#109028 work-around, however join+shell_quote probably did the right thing:
  $ldflags =~ s/^\s+|\s+$//g; 

  return $ldflags;
}

# The original Devel::CheckLib::check_lib() function tests libraries, but there
# are several noteworthy deficiencies (bugs on RT) to be aware of:
#
#   RT#60176: (12 years ago) no way to test with dependent libraries
#   RT#75803: (10 years ago) Allow function option to be used without lib option
#   RT#109028: (7 years ago) Problems with trailing space in ldflags
#
# Given the age of these bugs, I don't expect a fix any time soon.
#  
# Our new_check_lib() attempts to work around these problems as follows.  Maybe
# a new CPAN module should be released for this, but that sounds like a bit too
# much maintenance...anyone up for that?  Ahem.  Anyway:
#
#   To address #60176: No way to test with dependent libraries
#      The issue here is that check_lib() always tests libraries one at a time
#      instead of trying to test them together. To solve this, we only pass the
#      first library to `lib=>`; all remaining libs are passed as ldflags.
#      This is necessary because all dependent libs must be on the cmdline to
#      work together (like blas+lapack). Note that `lib=>` must be defined or
#      check_lib() will silently _NOT_ run the `function=>$f` argument!
#
#      In this implementation we assume all libs are dependent, so call
#      new_check_lib iteratively if you want to test one at a time.
#     
#   To address 109028#: Problems with trailing space in ldflags
#      $ldflags =~ s/^\s+|\s+$//g;
#
#   To address 75803#: Allow function option to be used without lib option
#      This would be nice to fix, but we don't really need it for the
#      PDL::LinearAlgebra module.  In our case we _always_ have a library to
#      test or we wouldn't call the function.
#
#      However, should someone ever need to evaluate a C `function=>` without
#      trying a library link, then one could pass `lib=>'c'` to check_lib()
#      hoping that everyone has a libc library.  (This probably doesn't work in
#      windows, so maybe it should pass kernel32.dll?)
#
#   In addition to those above:
#     We have found that check_lib()'s `libpath=>` option will (sometimes?)
#     place the -L options _after_ the -l options, possibly causing the
#     compiler to fail to find the -l<lib>.  To address this we pass -L options
#     on the ldflags line instead of `libpath=>`.
#
# Even in the midst of these deficiencies, Devel::CheckLib does a good job
# with the intracacies of different compilers, so no sense re-inventing it.
# The original check_lib() funciton is the core of this wrapper.
#
# Note that new_check_lib() parses the 'lib', 'libpath', and 'ldflags'
# arguments so 'ldflags' is the only necessary argument.  If all or any of
# these are supplied then they will be merged.
sub new_check_lib
{
  my (%args) = @_;

  my (@libs, @dirs, @ld_other);

  # Take a single lib or a arrayref of _dependent_ libs:
  @libs = @{ $args{lib} } if defined($args{lib}) && ref($args{lib}) eq 'ARRAY';
  push @libs, $args{lib} if defined($args{lib}) && !ref($args{lib});

  # Separate libs, dirs, and other options:
  if (defined($args{ldflags})) {
    my $ldflags = delete $args{ldflags};
    push @libs, _l($ldflags);
    push @dirs, _L($ldflags);
    push @ld_other, _ld_other($ldflags);
  }

  # work around possible libpath -L re-ordering by moving libpath
  # to @dirs:
  if (defined($args{libpath})) {
    push @dirs, @{ delete $args{libpath} }
  }

  # Split out dependent libs in to first and other (RT#60176 workaround):
  my ($first_lib, @other_libs) = @libs;
  $args{lib} = $first_lib;

  warn "No lib passed and check_lib will return a (possibly) false success" if !$args{lib};

  # rebuild $ldflags placing @dirs in front of @other_libs.
  $args{ldflags} = _merge_ld_flags(\@dirs, \@other_libs, \@ld_other);

  print STDERR "check_lib($args{lib}):  $args{ldflags}\n" if $ENV{PDL_LA_DEBUG};

  return check_lib(%args);
}

# Filter $ldflags and return $ldflags only with libs that compile successfully.
# This only filters -l<lib> libraries, so directories and other flags are
# passed verbatim.
sub filter_libs
{
  my $ldflags = shift;

  my @libs = _l($ldflags);
  my @dirs = _L($ldflags);
  my @ld_other = _ld_other($ldflags);

  my @goodlibs;
  foreach my $l (@libs)
  {
    if (!new_check_lib(ldflags => _merge_ld_flags(\@dirs, [ $l ], \@ld_other)))
    {
      warn "filter_libs: libary -l$l was not found, excluding."
    }
    else
    {
      push @goodlibs, $l
    }
  }

  return _merge_ld_flags(\@dirs, \@goodlibs, \@ld_other);
}

sub find_so
{
  my ($dirs, $libs) = @_;

  my @found;

  my %files;
  foreach my $d (@$dirs)
  {
    opendir(my $dh, $d) || die "opendir: $d: $!";
    $files{$d} = [ sort grep { -f "$d/$_" && /\.so\.\d+$/ } readdir($dh) ];
  }

  foreach my $l (@$libs)
  {
    my $found_lib;
    foreach my $d (@$dirs)
    {
      my @possible_libs = map { "$d/$_" } grep { /^lib\Q$l\E\.so/ } @{ $files{$d} };

      next if (!@possible_libs);

      print STDERR "possible $l path: @possible_libs\n" if $ENV{PDL_LA_DEBUG};

      # Keep the first occurance:
      push @found, $possible_libs[0];
      $found_lib = 1;
      last;
    }

    # Keep the original if we didn't find a .so:
    push @found, $l if (!$found_lib);
  }

  return \@found;
}

sub find_libs
{
  my @args = @_;
  my $existing_ldflags = shell_quote(map { shellwords($_) } @args);

  # Test each library that was passed and make sure it compiles:
  my $existing_ldflags = filter_libs($existing_ldflags);

  return $existing_ldflags if $^O !~ /linux/i;

  my @existing_dirs = _L($existing_ldflags);
  my @existing_libs = _l($existing_ldflags);
  my @existing_ld_other = _ld_other($existing_ldflags);

  # In performance order based on libraries I've used
  # for xnec2c in Ubuntu, Debian, SuSE, CentOS, etc.
  # See comments here for detail:
  #    https://github.com/KJ7LNW/xnec2c/blob/master/src/mathlib.c#L29
  my @libsets = (
      # Empty case if PDL_LA_LDFLAGS satisfies the build
      [],

      # Threaded OpenBLAS/ATLAS
      ['openblaso'],
      ['openblas_openmp'],
      ['tatlas'],
      ['openblasp'],
      ['openblas_pthreads'],

      # Serial OpenBLAS/ATLAS
      ['openblas'],
      ['openblas_serial'],
      ['satlas'],
      ['atlas'],

      # Remaining Guesses
      ['lapack'],
      ['blas'],
      ['lapack', 'blas'],
      ['lapack', 'blas', 'atlas'],
      ['lapack_atlas'],

      # Intel Math Kernel Library
      #
      # Set one of these environment variables to select a different threading model:
      #   MKL_THREADING_LAYER=SEQUENTIAL
      #   MKL_THREADING_LAYER=INTEL
      #   MKL_THREADING_LAYER=GNU
      #   MKL_THREADING_LAYER=PGI
      # (see page 30:
      #   https://www.intel.com/content/dam/develop/external/us/en/documents/onemkl-developerguide-linux.pdf )
      #
      # Setting this variable at configure time isn't critical, but you will want
      # to choose your desired threading model at runtime in your
      # PDL::LinearAlgebra application:
      #
      #   ~] export MKL_THREADING_LAYER=INTEL ; ./my-super-awesome-benchmark.pl
      #
      ['mkl_rt'],
    );

  printf STDERR "Checking LAPACK libraries...\n";

  my @found;
  my $found_idx = -1;
  for my $ls (@libsets) {

    if ($ENV{PDL_LA_DEBUG})
    {
      my $sep = '=' x 80;
      print STDERR "\n$sep\n";
      my $name = "@$ls" || '(default)';
      print STDERR "== Testing library set: $name\n";
      print STDERR "$sep\n";
    }

    # Run pkgconfig on each lib individually so if pkgconfig returns
    # an empty list then -l<name> will be used:
    my (@libs, @dirs, @ld_other);
    foreach my $l (@$ls) {
      my ($pc, @pclibs);

      # Threaded/serial ATLAS on RHEL is special:
      if ($l =~ /[st]atlas/) {
        $pc = pkgconfig('atlas');
        @pclibs = ($l);
      }
      else {
        $pc = pkgconfig($l);

        @pclibs = _l($pc);

        # If pkgconfig does not return any libs, then specify the -l<name> itself:
        push @pclibs, $l if !@pclibs;
      }

      push @libs, @pclibs;
      push @dirs, _L($pc);
      push @ld_other, _ld_other($pc);
    }

    # Existing libs must go at the end of the list or CheckDir will put the
    # LAPACK libs we are testing before the .c file in the compile string and
    # libraries will be missing:
    push @libs, @existing_libs;
    push @dirs, @existing_dirs;
    push @ld_other, @existing_ld_other;

    die "BUG: \@libs is empty?" if !@libs;

    # Try the -l<lib> version:
    my $ldflags = _merge_ld_flags(\@dirs, \@libs, \@ld_other);

    # Also try to find lib<lib>.so.X files in the -L search path; this works
    # for older debian/ubuntu builds:
    my $ldflags_so = _merge_ld_flags(\@dirs, find_so(\@dirs, \@libs),
	    \@ld_other);

    my %common_args = (
           debug => $ENV{PDL_LA_DEBUG},
           header => [ ( $^O =~ /MSWin/ ? 'float.h' : () ), qw(stdio.h math.h) ],
           function => $fortran_checklib);

    if (new_check_lib(%common_args, ldflags => $ldflags)) {
      $found_idx++;
      print STDERR "  [$found_idx] found (@$ls): $ldflags\n";

      push @found, $ldflags;
    }
    elsif (new_check_lib(%common_args, ldflags => $ldflags_so)) {
      $found_idx++;
      print STDERR "  [$found_idx] found (@$ls): $ldflags_so\n";

      push @found, $ldflags_so;
    }
    else
    {
      print STDERR "      missing (@$ls)\n" if @$ls;
    }
  }

  die "ERROR: Cannot find LAPACK library that links without error!\n" if !@found;

  my $selected_lib_idx = $ENV{PDL_LA_LIB_INDEX} || 0;
  
  my $selected_ldflags = $found[$selected_lib_idx];

  print STDERR qq{
    
If you prefer a different LAPACK implementation you can select it by:

1. Specifying PDL_LA_LDFLAGS that are sufficient to run FORTRAN LAPACK functions:
   ~# PDL_LA_LDFLAGS='-lsomething -L/somewhere -lelse' perl $0
2. Specifying the index shown above like '[2] found (...)' as follows:
   ~# PDL_LA_LIB_INDEX=2 perl $0

Distributions like Debian and Ubuntu may be configured using `alternatives`:
  ~# update-alternatives --config libblas.so.3-x86_64-linux-gnu
  ~# update-alternatives --config liblapack.so.3-x86_64-linux-gnu
You may need to re-run `perl $0` after changing your LAPACK selection with
alternatives.

You may also specify PDL_LA_DEBUG=1 to enable library detection debugging,
however this is very noisy.

PDL::LinearAlgebra will use these LAPACK libraries (index $selected_lib_idx):
  $selected_ldflags
};

  print STDERR "\nNOTICE: Intel MKL Library support is experimental!\n\n" if ($selected_ldflags =~ /mkl_/);

  return $selected_ldflags;
}

