function cost = CalcFidelityCost( est, coil_sense, k_space, sample_pattern )
	cost = repmat( est, [1 1 size(k_space,3) 1] );
	cost = cost .* coil_sense;
	cost = fft2( cost );
	cost = cost .* sample_pattern;
	cost = cost - k_space;
	cost = SSQ( cost );
