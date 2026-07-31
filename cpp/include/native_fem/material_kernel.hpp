#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace native_fem {

// Stateful small-strain material contract.  A kernel used by the global
// assembler provides `state_size` and a statically dispatched `update` with
// the same arguments as J2MaterialKernel::update below.
struct J2PointResult {
    std::array<double,6> stress{};
};

class J2MaterialKernel {
public:
    static constexpr int state_size=7;
    using Result=J2PointResult;

    J2MaterialKernel(double young,double poisson,double yield_stress,
                     double hardening)
        :young_(young),poisson_(poisson),yield_(yield_stress),
         hardening_(hardening) {
        if(young<=0.||poisson<=-1.||poisson>=.5||yield_stress<0.
           ||hardening<0.)
            throw std::invalid_argument("invalid J2 material parameters");
    }

    J2PointResult update(const double*strain,const double*committed,
                         double*trial_state,double*tangent=nullptr,
                         double=0.)const {
        const double mu=young_/(2.*(1.+poisson_));
        const double lambda=young_*poisson_/
            ((1.+poisson_)*(1.-2.*poisson_));
        std::array<double,6> elastic{},trial{},deviator{};
        for(int i=0;i<6;++i)
            elastic[i]=strain[i]-committed[i];
        const double trace=elastic[0]+elastic[1]+elastic[2];
        for(int i=0;i<3;++i)trial[i]=lambda*trace+2.*mu*elastic[i];
        for(int i=3;i<6;++i)trial[i]=2.*mu*elastic[i];
        const double mean=(trial[0]+trial[1]+trial[2])/3.;
        deviator=trial;
        for(int i=0;i<3;++i)deviator[i]-=mean;
        const double equivalent=std::sqrt(1.5*inner(deviator,deviator));
        const double current_yield=
            yield_+hardening_*committed[6];
        const bool plastic=equivalent>current_yield&&equivalent>1e-30;
        const double increment=plastic?
            (equivalent-current_yield)/(3.*mu+hardening_):0.;
        const double scale=plastic?1.-3.*mu*increment/equivalent:1.;

        J2PointResult result;
        std::copy(committed,committed+state_size,trial_state);
        result.stress=deviator;
        for(double&value:result.stress)value*=scale;
        for(int i=0;i<3;++i)result.stress[i]+=mean;
        if(plastic) {
            for(int i=0;i<6;++i)
                trial_state[i]+=increment*1.5*deviator[i]/equivalent;
            trial_state[6]+=increment;
        }
        if(tangent)
            consistent_tangent(
                deviator,equivalent,current_yield,plastic,tangent
            );
        return result;
    }

private:
    static double inner(const std::array<double,6>&a,
                        const std::array<double,6>&b) {
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
            +2.*(a[3]*b[3]+a[4]*b[4]+a[5]*b[5]);
    }

    void consistent_tangent(const std::array<double,6>&trial_deviator,
                            double equivalent,double current_yield,
                            bool plastic,double*out)const {
        const double mu=young_/(2.*(1.+poisson_));
        const double bulk=young_/(3.*(1.-2.*poisson_));
        double scale=1.,directional=0.;
        if(plastic) {
            const double denominator=3.*mu+hardening_;
            scale=hardening_/denominator+
                3.*mu*current_yield/(denominator*equivalent);
            directional=9.*mu*mu*current_yield/
                (denominator*equivalent*equivalent*equivalent);
        }
        for(int column=0;column<6;++column) {
            std::array<double,6> increment{};increment[column]=1.;
            const double trace=increment[0]+increment[1]+increment[2];
            std::array<double,6> deviator=increment;
            for(int i=0;i<3;++i)deviator[i]-=trace/3.;
            const double projection=inner(trial_deviator,deviator);
            for(int row=0;row<6;++row) {
                const double volumetric=row<3?bulk*trace:0.;
                out[row*6+column]=volumetric+2.*mu*scale*deviator[row]
                    -directional*trial_deviator[row]*projection;
            }
        }
    }

    double young_,poisson_,yield_,hardening_;
};

class LinearElasticMaterialKernel {
public:
    static constexpr int state_size=0;

    LinearElasticMaterialKernel(double young,double poisson)
        :young_(young),poisson_(poisson) {
        if(young<=0.||poisson<=-1.||poisson>=.5)
            throw std::invalid_argument("invalid linear elastic parameters");
    }

    J2PointResult update(const double*strain,const double*,double*,
                         double*tangent=nullptr,double=0.)const {
        const double mu=young_/(2.*(1.+poisson_));
        const double lambda=young_*poisson_/
            ((1.+poisson_)*(1.-2.*poisson_));
        const double trace=strain[0]+strain[1]+strain[2];
        J2PointResult result;
        for(int i=0;i<3;++i)
            result.stress[i]=lambda*trace+2.*mu*strain[i];
        for(int i=3;i<6;++i)result.stress[i]=2.*mu*strain[i];
        if(tangent) {
            std::fill(tangent,tangent+36,0.);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)
                tangent[6*i+j]=lambda+(i==j?2.*mu:0.);
            tangent[6*3+3]=tangent[6*4+4]=tangent[6*5+5]=2.*mu;
        }
        return result;
    }

private:
    double young_,poisson_;
};

class StandardLinearSolidKernel {
public:
    static constexpr int state_size=6;

    StandardLinearSolidKernel(double equilibrium_modulus,double branch_modulus,
                              double poisson,double relaxation_time,
                              double time_step)
        :equilibrium_(equilibrium_modulus),branch_(branch_modulus),
         poisson_(poisson),relaxation_time_(relaxation_time),
         time_step_(time_step) {
        if(equilibrium_modulus<=0.||branch_modulus<0.||poisson<=-1.||
           poisson>=.5||relaxation_time<=0.||time_step<=0.)
            throw std::invalid_argument(
                "invalid standard linear solid parameters"
            );
    }

    J2PointResult update(const double*strain,const double*committed,
                         double*trial_state,double*tangent=nullptr,
                         double evaluation_time_step=0.)const {
        const double dt=evaluation_time_step>0.?evaluation_time_step:time_step_;
        const double factor=1./(1.+dt/relaxation_time_);
        for(int i=0;i<6;++i)
            trial_state[i]=factor*committed[i]+(1.-factor)*strain[i];
        J2PointResult result;
        apply_elastic(equilibrium_,strain,result.stress.data());
        std::array<double,6>branch_strain{},branch_stress{};
        for(int i=0;i<6;++i)branch_strain[i]=strain[i]-trial_state[i];
        apply_elastic(branch_,branch_strain.data(),branch_stress.data());
        for(int i=0;i<6;++i)result.stress[i]+=branch_stress[i];
        if(tangent)elastic_tangent(equilibrium_+factor*branch_,tangent);
        return result;
    }

private:
    void apply_elastic(double young,const double*strain,double*stress)const {
        const double mu=young/(2.*(1.+poisson_));
        const double lambda=young*poisson_/
            ((1.+poisson_)*(1.-2.*poisson_));
        const double trace=strain[0]+strain[1]+strain[2];
        for(int i=0;i<3;++i)stress[i]=lambda*trace+2.*mu*strain[i];
        for(int i=3;i<6;++i)stress[i]=2.*mu*strain[i];
    }

    void elastic_tangent(double young,double*tangent)const {
        const double mu=young/(2.*(1.+poisson_));
        const double lambda=young*poisson_/
            ((1.+poisson_)*(1.-2.*poisson_));
        std::fill(tangent,tangent+36,0.);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            tangent[6*i+j]=lambda+(i==j?2.*mu:0.);
        tangent[6*3+3]=tangent[6*4+4]=tangent[6*5+5]=2.*mu;
    }

    double equilibrium_,branch_,poisson_,relaxation_time_,time_step_;
};

}  // namespace native_fem
