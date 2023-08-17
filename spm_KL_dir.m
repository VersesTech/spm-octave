function [d] = spm_KL_dir(q,p)
    % KL divergence between two Dirichlet distributions
    % FORMAT [d] = spm_kl_dir(lambda_q,lambda_p)
    %
    % Calculate KL(Q||P) = <log Q/P> where avg is wrt Q between two Dirichlet 
    % distributions Q and P
    %
    % lambda_q   -   concentration parameter matrix of Q
    % lambda_p   -   concentration parameter matrix of P
    %
    % This routine uses an efficient computation that handles arrays, matrices 
    % or vectors. It returns the sum of divergences over columns.
    %
    % See also: spm_kl_dirichlet.m (for row vectors)
    %__________________________________________________________________________
    
    % Karl Friston 
    % Copyright (C) 2018-2022 Wellcome Centre for Human Neuroimaging
    
    
    % KL divergence based on log beta functions
    %--------------------------------------------------------------------------
    spm_psi = @(q) minus(psi(q),psi(sum(q)));
    
    p = max(p,exp(-16));
    q = max(q,exp(-16));
    d = spm_betaln(p) - spm_betaln(q) + sum((q - p).*spm_psi(q));
    d = sum(d(:));
    
    return
    
    % NOTES: numerical check on KL of Dirichlet distributions
    %==========================================================================
    p  = rand(6,1);
    q  = rand(6,1) + 1;
    p0 = sum(p);
    q0 = sum(q);
    
    d  = q - p;
    q0 = sum(q);
    p0 = sum(p);
    KL = spm_betaln(p) - spm_betaln(q) + d'*spm_psi(q)
    KL = gammaln(q0)   - sum(gammaln(q)) - gammaln(p0) + sum(gammaln(p)) + ...
        d'*(spm_psi(q) - spm_psi(q0))
    
    KL = gammaln(sum(q)) - sum(gammaln(q)) + ...
    sum(gammaln(p)) - gammaln(sum(p)) + (q - p)'*(psi(q) - psi(sum(q)))
    