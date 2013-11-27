use ExtUtils::MakeMaker;

WriteMakefile(
        'NAME' => 'PDL::LinearAlgebra',
        'VERSION' => '0.06_02',
        'CONFIGURE_REQUIRES' =>  { "PDL" => 0, },
        'BUILD_REQUIRES' =>  { "PDL" => 0, },
	'PREREQ_PM' => { "PDL" => 0, },
        'DIR' =>  [ qw/Trans Real Complex Special/],
       );
__END__

