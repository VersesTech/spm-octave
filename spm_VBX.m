function [P,F] = spm_VBX(O,P,A,id)
    % vvariational Bayes estimate of categorical posterior over factors
    % FORMAT [Q,F] = spm_VBX(O,P,A,id)
    %
    % O{g}    -  outcome probabilities over each of G modalities
    % P{f}    -  (empirical) prior over each of F factors
    % A{g}    -  likelihood tensor for modality g
    %
    % Q{f}    -  variational posterior for each of F factors
    % F       -  (-ve)  variational free energy or ELBO
    %
    % This routine is a simple implementation of variational Bayes for discrete
    % state space models under a mean field approximation, in which latent
    % states are partitioned into factors (and the distribution over outcomes
    % is also assumed to be conditionally independent). It takes cell arrays of
    % outcome probabilities, prior probabilities over factors and a likelihood
    % tensor parameterising the likelihood of an outcome for any combination
    % of latent states. The optional argument METHOD [default: full] switches
    % among number of approximate schemes:
    %
    % 'full'    :  a vanilla variational scheme that uses a coordinate descent
    % over a small number (8) iterations (fixed point iteration). This is
    % computationally efficient, using only nontrivial posterior probabilities
    % and the domain of the likelihood mapping in tensor operations.
    % 
    %
    % 'exact'   :  a non-iterative heuristic but numerically accurate scheme
    % that replaces the variational density over hidden factors with the
    % marginal over the exact posterior
    %
    % 'sparse'  :  as for the exact scheme but suitable for sparse tensors
    %
    % 'marginal':  a heuristic scheme  that uses the log of the marginalised
    % likelihood and log prior to estimate the log posterior
    %
    % see: spm_MDP_VB_XXX.m (NOTES)
    %__________________________________________________________________________
    
    % Karl Friston
    % Copyright (C) 2012-2022 Wellcome Centre for Human Neuroimaging
    
    
    % preliminaries
    %--------------------------------------------------------------------------
    TOL    = 16;
    METHOD = 'full';
    
    switch METHOD
    
        case 'full'
    
            %  (iterative) variational scheme
            %==================================================================
    
            % log prior
            %------------------------------------------------------------------
            for f = 1:numel(P)
                i{f}  = find(P{f} > exp(-8));
                Q{f}  = P{f}(i{f});
                LP{f} = spm_vec(spm_log(Q{f}));
            end
    
            % accumulate log likelihoods over modalities
            %------------------------------------------------------------------
            L     = 0 ;
            for g = 1:numel(O)
                j  = id.A{g};
                LL = spm_log(spm_dot(A{g}(:,i{j}),O{g}));
                
                k  = ones(1,8); k(j) = size(LL,1:numel(j));
                L  = plus(L,reshape(LL,k));
            end
    
            % preclude numerical overflow of log likelihood
            %------------------------------------------------------------------
            L = max(L, max(L(:)) - TOL);
    
    
            % variational iterations
            %------------------------------------------------------------------
            Z     = -Inf;
            for v = 1:8
                F     = 0;
                for f = 1:numel(P)
    
                    % log likelihood
                    %----------------------------------------------------------
                    LL   = spm_vec(spm_dot(L,Q,f));
    
                    % posterior
                    %----------------------------------------------------------
                    Q{f} = spm_softmax(LL + LP{f});
    
                    % (-ve) free energy (partition coefficient)
                    %----------------------------------------------------------
                    F    = F + Q{f}'*(LL + LP{f} - spm_log(Q{f}));
                end
    
                % convergence
                %--------------------------------------------------------------
                if F > 0
                    %%% warning('positive ELBO in spm_VBX')
                end
                dF = F - Z;
                if dF < 1/128
                    break
                elseif dF < 0
                    warning('ELBO decreasing in spm_VBX')
                else
                    Z = F;
                end
            end
    
            % Posterior
            %------------------------------------------------------------------
            for f = 1:numel(P)
                P{f}(:)    = 0;
                P{f}(i{f}) = Q{f};
            end
    
    
        case 'exact'
    
            % belief propagation with marginals of exact posterior
            %==================================================================
    
            % prior
            %------------------------------------------------------------------
            for f = 1:numel(P)
                i{f}  = find(P{f} > exp(-8));
                R{f}  = P{f}(i{f});
                Ns(f) = numel(P{f});
            end
    
            % accumulate likelihoods over modalities
            %------------------------------------------------------------------
            L     = 1;
            for g = 1:numel(O)
                j  = id.A{g};
                LL = spm_dot(A{g}(:,i{j}),O{g});
                
                k  = ones(1,8); k(j) = size(LL,1:numel(j));
                L  = times(L,reshape(LL,k));
            end
            
            % marginal posteriors and free energy (partition function)
            %------------------------------------------------------------------
            U     = L.*spm_cross(R);                   % posterior unnormalised
            Z     = sum(U,'all');                      % partition coefficient
            if Z
                F = spm_log(Z);                        % negative free energy
                Q = spm_marginal(U/Z);                 % marginal  posteriors
            else
                F = -32;                               % negative free energy
                Q = spm_marginal(U + 1/numel(U));      % marginal  posteriors
            end
    
            % (-ve) free energy (partition coefficient)
            %------------------------------------------------------------------
            F     = 0;
            for f = 1:numel(P)
               LL = spm_vec(spm_dot(spm_log(L),Q,f));
               LP = spm_vec(spm_log(R{f}));
               F  = F + Q{f}'*(LL + LP - spm_log(Q{f}));
            end
    
            % Posterior
            %------------------------------------------------------------------
            for f = 1:numel(P)
                P{f}(:)    = 0;
                if numel(i{f}) > 1
                    P{f}(i{f}) = Q{f};
                else
                    P{f}(i{f}) = 1;
                end
            end
    
        case 'sparse'
    
            % approximation with marginals suitable for sparse tensors
            %==================================================================
            Nf    = size(A{1});
            L     = 1;
            for g = 1:numel(O)
                L = L.*(O{g}'*A{g}(:,:));              % likelihood over modalities
            end
            U     = spm_vec(L).*spm_vec(spm_cross(P)); % posterior unnormalised
            Z     = sum(U,'all');                      % partition coefficient
            F     = spm_log(Z);                        % negative free energy 
            U     = reshape(U/Z,[Nf(2:end),1]);        % joint posterior
            Q     = spm_marginal(U);                   % marginal  posteriors
    
        otherwise
            disp('unknown method')
    
    end
    
    return
