use strict;
use warnings;
use PDL::LiteF;
use PDL::MatrixOps qw(identity);
use PDL::LinearAlgebra;
use PDL::LinearAlgebra::Trans qw //;
use PDL::LinearAlgebra::Real;
use PDL::Complex;
use Test::More;

sub fapprox {
	my($a,$b) = @_;
	($a-$b)->abs->max < 0.0001;
}
sub runtest {
  local $Test::Builder::Level = $Test::Builder::Level + 1;
  my ($in, $method, $expected, $extra) = @_;
  ($expected, my $expected_cplx) = ref($expected) eq 'ARRAY' ? @$expected : ($expected, $expected);
  my ($got) = $in->$method(@{$extra||[]});
  ok fapprox($got, $expected), $method or diag "got: $got";
  $_ = PDL::Complex::r2C($_) for $in, $expected_cplx;
  ($got) = $in->$method(map ref() ? PDL::Complex::r2C($_) : $_, @{$extra||[]});
  ok fapprox($got, $expected_cplx), "PDL::Complex $method" or diag "got: $got";
}

my $a = pdl([[1.7,3.2],[9.2,7.3]]);
runtest($a, 't', $a->xchg(0,1));

my $aa = cplx random(2,2,2);
runtest($aa, 't', $aa->xchg(1,2), [0]);

runtest(sequence(2,2), 'issym', 0);

my $x = pdl([0.43,0.03],[0.75,0.72]);
runtest($x, 'mschur', pdl([0.36637354,-0.72],[0,0.78362646]));
runtest(sequence(2,2), 'diag', pdl(0,3));
runtest(sequence(3,3), 'tritosym', pdl [0,1,2],[1,4,5],[2,5,8]);
runtest(pdl([1,2],[1,0]), 'mrcond', 1/3);
runtest($x, 'mtriinv', pdl([2.3255814,-0.096899225],[0.75,1.3888889]));
runtest($x, 'msyminv', pdl([2.3323615,-0.09718173],[-0.09718173,1.3929381]));
runtest($x->crossprod($x), 'mchol', pdl([0.86452299,0.63954343],[0,0.33209065]));
my @mgschur_exp = (pdl([-0.35099581,-0.68880032],[0,0.81795847]),
  pdl([1.026674, -0.366662], [0, -0.279640]));
runtest($x, 'mgschur', \@mgschur_exp, [sequence(2,2)]);
runtest($x, 'mgschurx', \@mgschur_exp, [sequence(2,2)]);
runtest($x, 'mqr', pdl([-0.49738411,-0.86753043],[-0.86753043,0.49738411]));
runtest($x, 'mrq', pdl([0.27614707,-0.3309725],[0,-1.0396634]));
runtest($x, 'mql', pdl([0.99913307,-0.041630545],[-0.041630545,-0.99913307]));
runtest($x, 'mlq', pdl([-0.43104524,0],[-0.79829207,0.66605538]));
runtest($x, 'msolve', pdl([-0.20898642,2.1943574],[2.995472,1.8808777]), [sequence(2,2)]);
runtest($x, 'mtrisolve', pdl([0,2.3255814],[2.7777778,1.744186]), [1,sequence(2,2)]);
runtest($x, 'msymsolve', pdl([5.9311981,6.0498221],[-3.4005536,-2.1352313]), [1,sequence(2,2)]);
runtest(pdl([2,-1,0],[-1,2,-1],[0,-1,2]), 'mpossolve', pdl([3,4.5,6],[6,8,10],[6,7.5,9]), [1,sequence(3,3)]);
runtest($x, 'mglm', pdl([-0.10449321,1.497736],[30.95841,-44.976237]), [sequence(2,2),sequence(2,2)]);
runtest($x, 'meigen', pdl([0.366373539549749,0.783626460450251]));
runtest($x, 'mgeigen', [pdl([-0.350995,0.817958]), pdl([1.026674,-0.279640])], [sequence(2,2)]);
runtest($x, 'msymeigen', pdl([0.42692907,0.72307093]));
runtest($x, 'mdsvd', pdl([0.32189374,0.9467758],[0.9467758,-0.32189374]));
runtest($x, 'mgsvd', pdl(0.16914549,0.64159379), [sequence(2,2)]);
runtest($a, 'mdet', -17.03);
runtest($a->mcos, 'macos', pdl([[1.7018092, 0.093001244],[0.26737858,1.8645614]]));
runtest($a->msin, 'masin', pdl([[ -1.4397834,0.093001244],[0.26737858,-1.2770313]]));
runtest($a->mexp, 'mlog', $a);

my $id = pdl([[1,0],[0,1]]);
ok(fapprox($a->minv x $a,$id));

ok(fapprox($a->mcrossprod->mposinv->tritosym x $a->mcrossprod,$id));

ok($a->mcrossprod->mposdet !=0);

my $A = identity(4) + ones(4, 4);
$A->slice('2,0') .= 0; # if don't break symmetry, don't show need transpose
my $B = sequence(2, 4);
getrf(my $lu=$A->copy, my $ipiv=null, my $info=null);
# if don't transpose the $B input, get memory crashes
getrs($lu, 1, $x=$B->xchg(0,1)->copy, $ipiv, $info=null);
$x = $x->inplace->xchg(0,1);
my $got = $A x $x;
ok fapprox($got, $B) or diag "got: $got";

$A=pdl cdouble, <<'EOF';
[
 [  1   0   0   0   0   0]
 [0.5   1   0 0.5   0   0]
 [0.5   0   1   0   0 0.5]
 [  0   0   0   1   0   0]
 [  0   0   0 0.5   1 0.5]
 [  0   0   0   0   0   1]
]
EOF
PDL::LinearAlgebra::Complex::cgetrf($lu=$A->copy, $ipiv=null, $info=null);
is $info, 0, 'cgetrf native worked';
is $ipiv->nelem, 6, 'cgetrf gave right-sized ipiv';
$B=pdl q[0.233178433563939+0.298197173371207i 1.09431208340166+1.30493506686269i 1.09216041861621+0.794394153882734i 0.55609433247125+0.515431151337765i 0.439100406078467+1.39139453403467i 0.252359761958406+0.570614019329113i];
PDL::LinearAlgebra::Complex::cgetrs($lu, 1, $x=$B->copy, $ipiv, $info=null);
is $info, 0;
$x = $x->dummy(0); # transpose; xchg rightly fails if 1-D
$got = $A x $x;
ok fapprox($got, $B->dummy(0)) or diag "got: $got";
my $i=pdl('i'); # Can't use i() as it gets confused by PDL::Complex's i()
my $complex_matrix=(1+sequence(2,2))*$i;
$got=$complex_matrix->mdet;
ok(fapprox($got, 2), "Complex mdet") or diag "got $got";

done_testing;
