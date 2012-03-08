function ssq_input = SSQ( ssq_input )
	ssq_input = abs(ssq_input).^2;
	dims = numel( size( ssq_input ) );
	for( k = 1:dims )
		ssq_input =  sum( ssq_input, k );
	end
	
