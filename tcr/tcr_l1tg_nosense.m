%function [est cost] = tcrl1tg_nosense( k_space, iterations, alpha, beta_sqrd, step_size )
k_space = cine_data(:,:,:,1,1:60);
iterations = 300;
alpha = 0.08;
beta_sqrd=0.00000000001;
step_size = 0.2;

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
est_orig = SendDat( gather( k_space ), 'sebulba', 9999, 'temp_interp_k' );


%for( curr_ch = 1:size(est_orig,3) )
for( curr_ch = 5 )

orig_im = IFFT2D( est_orig );
orig_im = abs( orig_im(:,:,5,1,:) );

est = gpuArray( est_orig(:,:,curr_ch,:,:) );

% iterate
k = 1;
while( k <= iterations )
	disp( k );

	% get fidelity term
	fidelity_term = est - k_space(:,:,curr_ch,:,:);
	fidelity_term = fidelity_term .* sample_pattern(:,:,1,:,:);

	% temporal gradient (l1)
    forward_time_imgs = circshift(est,[0 0 0 0 1]);
    backward_time_imgs = circshift(est,[0 0 0 0 -1]);
    term1_imgs=(forward_time_imgs-est)./(sqrt(beta_sqrd+(abs(forward_time_imgs-est).^2)));
    term2_imgs=(est - backward_time_imgs)./(sqrt(beta_sqrd+(abs(est-backward_time_imgs).^2)));
    temp_term=(term1_imgs-term2_imgs)*-0.5;

	% update
	gradient = fidelity_term + alpha.*alpha.*temp_term;
	est = est - step_size.*gradient;

	%isc( fftshift( ifft2( est ) ) );
	est_h = gather( est );
	est_h = IFFT2D( est_h );
	%isc( squeeze( est_h(191,:,1,1,:)));
	isc( cat( 2, ... 
		imnormalize( squeeze( est_h(191,:,1,1,:))), ...
		imnormalize( squeeze( orig_im(191,:,1,1,:))) ));
	drawnow;
	k = k + 1;
end

est_all(:,:,curr_ch,:,:) = gather( est );
est = fftshift( fftshift( est, 1 ), 2 );

end
