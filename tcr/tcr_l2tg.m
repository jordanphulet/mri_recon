%function [est cost] = tcr( k_space, iterations, alpha, step_size )
k_space = cine_data(:,:,:,1,:);
iterations = 500;
alpha = 0.25;
step_size = 1;

[n_y, n_x, n_ch, n_set, n_ph] = size( k_space );

% prepare k_space
k_space = k_space ./ max( abs( k_space(:) ) );
k_space = fftshift( fftshift( k_space, 1 ), 2 );
k_space = gpuArray( k_space );

% calculate sampling pattern
sample_pattern = abs( k_space ) > 1e-20;

% generate coil sensitivities
est_prime = ifft2( sum( k_space, 5 ) );
ssq = sqrt( sum( abs( est_prime ).^2, 3 ) );
ssq = repmat( ssq, [1 1 n_ch 1] );
coil_sense = est_prime ./ ssq;
coil_sense = repmat( coil_sense, [1 1 1 1 n_ph] );
clear est_prime;
clear ssq;


% generate initial estimate
est = SendDat( gather( k_space ), 'sebulba', 9999, 'temp_interp_k' );
%est = sqrt( sum( abs( FFT2D( est ) ).^2, 3 ) );
est = sqrt( sum( abs( ifft( ifft( est, [], 1 ), [], 2 ) ).^2, 3 ) );
est = gpuArray( est );

% keep track of cost
cost = gather( ...
	CalcFidelityCost( est, coil_sense, k_space, sample_pattern ) + ...
	alpha .* CalcL2TGCost( est ) );
disp( cost );

est0 = gather( est );

% iterate
k = 1;
while( k <= iterations )
	disp( k );

	% split estimate into channels
	est_ch = repmat( est, [1 1 n_ch, 1] );

	% apply sensitivity
	est_ch = est_ch .* coil_sense;

	% FFT
	est_ch = fft2( est_ch );

	% under-sample
	est_ch = est_ch .* sample_pattern;

	% subtract measured data
	est_ch = est_ch - k_space;

	% IFFT
	est_ch = ifft2( est_ch );

	% apply conj sensitivity
	est_ch = est_ch .* conj( coil_sense );

	% sum channels
	sense_term = sum( est_ch, 3 );

	% temporal gradient
	temp_term = 2*est - circshift(est,[0 0 0 0 1]) - circshift(est,[0 0 0 0 -1]);

	% update
	gradient = sense_term + alpha.*alpha.*temp_term;
	est = est - step_size.*gradient;

	new_cost = gather( ...
		CalcFidelityCost( est, coil_sense, k_space, sample_pattern ) + ...
		alpha .* CalcL2TGCost( est ) );

	if( new_cost > cost(end) )
		est = est + step_size*gradient;
		step_size = step_size * 0.5;
		if( step_size < 1e-5 )
			break;
		end
		disp( sprintf( 'changing step size to %f...', step_size ) );
	else
		cost = [cost new_cost];
		disp( cost(end) );
		isc( fftshift( est ) );
		drawnow;
		k = k + 1;
	end
end

est = gather( est );
