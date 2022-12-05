#pragma once

#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <Eigen/Dense>

// #include "logger.h"

#include <memory>

class Controller
{
public:
    double * targetPose         = nullptr;
    double * targetPoseControl  = nullptr;
    double * error_position     = nullptr;
    double * jac_CoM            = nullptr;
    double * jac_LeftFoot       = nullptr;
    double * error_state        = nullptr;
    double * A                  = nullptr;
    double * B                  = nullptr;
    double * K                  = nullptr;
    double * Q_joint            = nullptr;
    double * Q                  = nullptr;
    double * Q_pos              = nullptr;
    double * P                  = nullptr;

    int32_t * actuated         = nullptr;
    double * deriv_mem         = nullptr;

    Eigen::Map<Eigen::MatrixXd> eInertiaMatrix;
    Eigen::Map<Eigen::MatrixXd> eMotorForceMatrix;
    Eigen::Map<Eigen::MatrixXd> eVelocityMatrix;
    Eigen::Map<Eigen::MatrixXd> eDesiredBodyForceMatrix;
    Eigen::Map<Eigen::MatrixXd> eOutputBodyForce;
    Eigen::Map<Eigen::MatrixXd> eTargetPoseControl;
    Eigen::Map<Eigen::MatrixXd> eError_position;
    Eigen::Map<Eigen::MatrixXd> eJac_CoM;
    Eigen::Map<Eigen::MatrixXd> eJac_LeftFoot;
    Eigen::Map<Eigen::MatrixXd> eError_state;
    Eigen::Map<Eigen::MatrixXd> eK;
    Eigen::Map<Eigen::MatrixXd> eA;
    Eigen::Map<Eigen::MatrixXd> eB;
    Eigen::Map<Eigen::MatrixXd> eQ_joint;
    Eigen::Map<Eigen::MatrixXd> eQ;
    Eigen::Map<Eigen::MatrixXd> eQpos;
    Eigen::Map<Eigen::MatrixXd> eP;
    
    Eigen::Map<Eigen::MatrixXd> eDqacc_Dctrl;

    Eigen::MatrixXd eAT;
    Eigen::MatrixXd eBT;
    Eigen::MatrixXd eRinv;
    Eigen::MatrixXd eR;
    Eigen::MatrixXd eQ_balance;

    Eigen::MatrixXd eInertiaMatrixSlice;
    Eigen::MatrixXd eBT_eP;
    Eigen::MatrixXd temp_eK;
    Eigen::MatrixXd eAT_eP;

    mjModel* m = nullptr;
    mjData*  d = nullptr;

    // std::unique_ptr<ILogger> positionLogger;
    // std::unique_ptr<ILogger> errorLogger;
    // std::unique_ptr<ILogger> actuationLogger;
    // std::unique_ptr<ILogger> controlLogger;

    Controller(mjData* data, mjModel* model);
    ~Controller();

    static
    void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src);

    void saveTargetPose();

    void loadTargetPose();

    void CalculateControlSetpoint();

    void CalculateUnderactuatedControlSetpoint();

    void CalculateLQR();

    void CalculateDerivative();

    void Setup();

    void loop(mjData* currentState);

    void applyControll(mjData* dst);
};

class ControllerFactory
{
public:
    static auto Create(mjData* data, mjModel* model) ->Controller
    {
        auto d = mj_makeData(model);
        Controller::cpMjData(model, d, data);

        return {d, model};
    }

    static auto CreateSharedPtr(mjData* data, mjModel* model) ->std::shared_ptr<Controller> 
    {
        auto d = mj_makeData(model);
        Controller::cpMjData(model, d, data);

        return std::make_shared<Controller>(d, model);
    }
};
