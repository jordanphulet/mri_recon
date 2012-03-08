function cost = CalcL2TGCost( est )
	cost = est - circshift( est, [0 0 0 0 1] );
	cost = SSQ( cost );
