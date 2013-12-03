use ExtUtils::MakeMaker;

WriteMakefile(
        'NAME' => 'PDL::LinearAlgebra',
        'VERSION_FROM' => 'LinearAlgebra.pm',
        'LICENSE' => 'artistic_2',
        'META_MERGE' => {
           resources => {
              homepage => 'http://pdl.perl.org/',
              repository  => {
                 url => 'git://git.code.sf.net/p/pdl/pdl-linearalgebra',
                 type => 'git',
                 web => 'http://sourceforge.net/p/pdl/pdl-linearalgebra/ci/master/tree/',
              },
           },
        },
        'CONFIGURE_REQUIRES' =>  { "PDL" => 0, },
        'BUILD_REQUIRES' =>  { "PDL" => 0, },
	'PREREQ_PM' => { "PDL" => 0, },
        'DIR' =>  [ qw/Trans Real Complex Special/],
       );
__END__

