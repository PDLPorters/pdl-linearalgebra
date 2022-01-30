use strict;
use warnings;
use PDL::LiteF;
use PDL::MatrixOps qw(identity);
use PDL::LinearAlgebra;
use PDL::LinearAlgebra::Trans qw //;
use PDL::LinearAlgebra::Complex;
use PDL::Complex;
use Test::More;

sub fapprox {
	my($a,$b) = @_;
	($a-$b)->abs->max < 0.0001;
}
# PDL::Complex only
sub runtest {
  local $Test::Builder::Level = $Test::Builder::Level + 1;
  my ($in, $method, $expected_cplx, $extra) = @_;
  $_ = PDL::Complex::r2C($_) for $in, $expected_cplx;
  my ($got) = $in->$method(map ref() ? PDL::Complex::r2C($_) : $_, @{$extra||[]});
  ok fapprox($got, $expected_cplx), "PDL::Complex $method" or diag "got: $got";
}

my $a = pdl([[1.7,3.2],[9.2,7.3]]);

my $aa = cplx random(2,2,2);
runtest($aa, 't', $aa->xchg(1,2));

runtest(sequence(2,2), 'issym', 0);

my $x = pdl([0.43,0.03],[0.75,0.72]);
runtest($x, 'mschur', pdl([0.36637354,-0.72],[0,0.78362646]));
runtest(sequence(2,2), 'diag', pdl(0,3));
runtest(sequence(3,3), 'tritosym', pdl [0,1,2],[1,4,5],[2,5,8]);
runtest(pdl([1,2],[1,0]), 'mrcond', 1/3);
runtest($x, 'mtriinv', pdl([2.3255814,-0.096899225],[0.75,1.3888889]));
runtest($x, 'msyminv', pdl([2.3323615,-0.09718173],[-0.09718173,1.3929381]));
runtest($x->crossprod($x), 'mchol', pdl([0.86452299,0.63954343],[0,0.33209065]));
runtest($x, 'mgschur', pdl([1.026674, -0.366662], [0, -0.279640]), [sequence(2,2)]);
runtest($x, 'mgschurx', pdl([1.026674, -0.366662], [0, -0.279640]), [sequence(2,2)]);
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
runtest($x, 'mgeigen', pdl([1.026674,-0.279640]), [sequence(2,2)]);
runtest($x, 'msymeigen', pdl([0.42692907,0.72307093]));
runtest($x, 'mdsvd', pdl([0.32189374,0.9467758],[0.9467758,-0.32189374]));
runtest($x, 'mgsvd', pdl(0.16914549,0.64159379), [sequence(2,2)]);
runtest($a, 'mdet', -17.03);
runtest($a->mcos, 'macos', pdl([[1.7018092, 0.093001244],[0.26737858,1.8645614]]));
runtest($a->msin, 'masin', pdl([[ -1.4397834,0.093001244],[0.26737858,-1.2770313]]));
runtest($a->mexp, 'mlog', $a);

done_testing;
