#include "controller.h"
#include "mujoco/mujoco.h"

#include "sstream"
#include <iostream>

// template <typename Derived>
// void LogEigen(std::unique_ptr<ILogger>& l, const Derived& b)
// {
//     std::stringstream ss3;
//     auto cols = b.cols();
//     auto colsM1 = cols - 1;
//     for(int j = 0; j < b.rows(); j++)
//     {
//         for(int i = 0; i < cols; i++)
//         {
//             ss3 << b.coeff(j,i);
//             if(colsM1 > i)
//             {
//                 ss3 << ",";
//             }
//             else
//             {
//                 ss3 << "\n";
//             }
//         }
//     }

//     l->Log(ss3.str());
// }

Controller::Controller(mjData * data, mjModel* model)
    :d                  (data)
    ,m                  (model)
    ,targetPose              (new double [model->nv])
    ,targetPoseControl       (new double [model->nu])
    ,error_position          (new double [model->nv])
    ,jac_CoM                 (new double [3*model->nv])
    ,jac_LeftFoot            (new double [3*model->nv])
    ,error_state             (new double [2*model->nv])
    ,A                       (new double [(2*model->nv)*(2*model->nv)])
    ,B                       (new double [(2*model->nv)*model->nu])
    ,K                       (new double [(2*model->nv)*model->nu])
    ,Q_joint                 (new double [model->nv*model->nv])
    ,Q                       (new double [(2*model->nv)*(2*model->nv)])
    ,Q_pos                   (new double [model->nv*model->nv])
    ,P                       (new double [(2*model->nv)*(2*model->nv)])
    ,actuated                (new int32_t[model->nv])
    ,eInertiaMatrix             (data->actuator_moment, model->nv, model->nu)
    ,eMotorForceMatrix          (data->ctrl, model->nu, 1)
    ,eVelocityMatrix            (data->qvel, 1, model->nv)
    ,eDesiredBodyForceMatrix    (data->qfrc_inverse, model->nv, 1)
    ,eOutputBodyForce           (data->qfrc_actuator, model->nv, 1)
    ,eTargetPoseControl         (this->targetPoseControl, model->nu, 1)
    ,eError_position            (this->error_position, 1, model->nv)
    ,eJac_CoM                   (this->jac_CoM, 3, model->nv)
    ,eJac_LeftFoot              (this->jac_LeftFoot, 3, model->nv)
    ,eError_state               (this->error_state, 1, (2*model->nv))
    ,eK                         (this->K, model->nu, (2*model->nv))
    ,eA                         (this->A, (2*model->nv), (2*model->nv))
    ,eB                         (this->B, (2*model->nv), model->nu)
    ,eQ                         (this->Q, (2*model->nv), (2*model->nv))
    ,eQpos                      (this->Q_pos, model->nv, model->nv)
    ,eQ_joint                   (this->Q_joint, model->nv, model->nv)
    ,eP                         (this->P, 2*model->nv, 2*model->nv)
{
    this->eQ_joint = Eigen::MatrixXd::Identity(m->nv, m->nv);
    this->eR = Eigen::MatrixXd::Identity(m->nu, m->nu);
    this->eP = Eigen::MatrixXd::Zero((2*model->nv), (2*model->nv));
    this->eK = Eigen::MatrixXd::Constant(m->nu, m->nv * 2, 0.01);

    for (uint32_t i = 0; i < m->nv; i++)
    {
        actuated[i] = -1;
    }

    // positionLogger = LoggerFactory::Create("position.csv");
    // errorLogger = LoggerFactory::Create("error.csv");
    // actuationLogger = LoggerFactory::Create("actuation.csv");
    // controlLogger = LoggerFactory::Create("control.csv");
};

Controller::~Controller()
{
    delete[] targetPose;
    delete[] targetPoseControl;
    delete[] error_position;
    delete[] jac_CoM;
    delete[] jac_LeftFoot;
    delete[] error_state;
    delete[] A;
    delete[] B;
    delete[] K;
    delete[] Q_joint;
    delete[] Q;
    delete[] Q_pos;
    delete[] P;
    delete[] actuated;
}

void Controller::cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src)
{
    d_dest->time = d_src->time;
    mju_copy(d_dest->qpos, d_src->qpos, m->nq);
    mju_copy(d_dest->qvel, d_src->qvel, m->nv);
    mju_copy(d_dest->qacc, d_src->qacc, m->nv);
    mju_copy(d_dest->qacc_warmstart, d_src->qacc_warmstart, m->nv);
    mju_copy(d_dest->qfrc_applied, d_src->qfrc_applied, m->nv);
    mju_copy(d_dest->xfrc_applied, d_src->xfrc_applied, 6*m->nbody);
    mju_copy(d_dest->ctrl, d_src->ctrl, m->nu);
}

void Controller::saveTargetPose()
{
    for(int i = 0; i < m->nv; i++)
    {
        targetPose[i] = d->qpos[i];
    }
}

void Controller::loadTargetPose()
{
    for(int i = 0; i < m->nv; i++)
    {
        d->qpos[i] = targetPose[i];
    }
}

void Controller::Setup()
{
    this->CalculateControlSetpoint();
    this->CalculateLQR();
}

void Controller::CalculateControlSetpoint()
{
    mj_resetDataKeyframe(m, d, 0);

    this->saveTargetPose();

    mj_forward(m, d);

    //==============================================================================================
    //                  Get joint forces that result in zero acceleration
    //==============================================================================================
    for (int i = 0; i <= m->nu; ++i)
    {
        d->qacc[i] = 0;
    }

    mj_inverse(m, d);

    //==============================================================================================
    //                          Get unactuated dimensions
    //==============================================================================================
    for (uint32_t i = 0; i < m->nv; i++)
    {
       for (uint32_t j = 0; j < m->nu; j++)
       {
           if(0 != eInertiaMatrix.coeff(i, j))
           {
                actuated[i] = i;
                break;
           }
       }
    }

    //==============================================================================================
    //                                  Calculate setpoints
    //==============================================================================================
    eMotorForceMatrix = eInertiaMatrix.fullPivLu().solve(eDesiredBodyForceMatrix);
 
    mj_forward(m, d);

    eTargetPoseControl = eMotorForceMatrix;

    //==============================================================================================
    //                          Compare generated forces with desired
    //==============================================================================================
    // #ifdef PRINTDEBUG
    std::cout << "Desired   " << eDesiredBodyForceMatrix.transpose() 
    << "\nActual    " << eOutputBodyForce.transpose()
    << "\nControl   " << eMotorForceMatrix.transpose() << std::endl;

    if((eDesiredBodyForceMatrix - eOutputBodyForce).isMuchSmallerThan(0.5))
    {
        std::cout << "\nFound a good setpoint" << std::endl;
    }
    else
    {
        std::cout << "\nDid not found a good setpoint, acceleration with this setpoint:" << std::endl;
        for (int i = 0; i <= m->nu; ++i)
        {
            std::cout << "\t" << d->qacc[i] << "\n";
        }
    }
    // #endif
}

void Controller::CalculateUnderactuatedControlSetpoint()
{
    auto const step = 0.01;
    //============================================================
    //                  Iterate max 100 times
    //============================================================
    for(int i = 0; i < 100; i++)
    {
        //============================================================
        //              Move controller by increment
        //============================================================
        this->CalculateDerivative();

        eDqacc_Dctrl = Eigen::MatrixXd::Constant(m->nu, m->nv, 0.01);
        eMotorForceMatrix = eMotorForceMatrix + (eDqacc_Dctrl * step);

        //============================================================
        //              Calculate new acc
        //============================================================
        mj_forward(m, d);

        //============================================================
        //              Break if control is good
        //============================================================
        if((eDesiredBodyForceMatrix - eOutputBodyForce).isMuchSmallerThan(0.5))
        {
            break;
        }
    }
}

void Controller::CalculateDerivative()
{

}

void Controller::CalculateLQR()
{
    //==============================================================
    //          LQR -> R
    //==============================================================
    //eR = Eigen::MatrixXd::Identity(m->nu, m->nu);

    #ifdef ISHUMANOID
    //==============================================================
    //          LQR -> Q for balance
    //==============================================================
    mj_jacSubtreeCom(m, d, jac_CoM, 1);

    mj_jacSubtreeCom(m, d, jac_LeftFoot, 10);

    auto jac_diff = eJac_CoM - eJac_LeftFoot;
    this->eQ_balance = jac_diff.transpose() * jac_diff; //eQ_balance (27 x 27)

    //==============================================================
    //          LQR -> Q
    //==============================================================
    // 0 root
    // 1 abdomen_z
    // 2 abdomen_y
    // 3 abdomen_x
    // 4 hip_x_right
    // 5 hip_z_right
    // 6 hip_y_right
    // 7 knee_right
    // 8 ankle_y_right
    // 9 ankle_x_right
    // 10 hip_x_left
    // 11 hip_z_left
    // 12 hip_y_left
    // 13 knee_left
    // 14 ankle_y_left
    // 15 ankle_x_left
    // 16 shoulder1_right
    // 17 shoulder2_right
    // 18 elbow_right
    // 19 shoulder1_left
    // 20 shoulder2_left
    auto root_dofs = {0, 1, 2, 3, 4, 5, 6};
    auto abdomen_dofs = {2, 3}; //[7, 8] online
    auto left_leg_dofs = {10, 12,  13,  14, 15}; //[15, 17, 18, 19, 20] online
    auto balance_dofs = {2, 3, 10, 12,  13,  14, 15};
    auto other_dofs = {0, 1, 4, 5, 6, 7, 8, 9, 11, 16, 17, 18, 19, 20};

    auto BALANCE_COST       = 1000;  // Balancing.
    auto BALANCE_JOINT_COST = 3;     // Joints required for balancing.
    auto OTHER_JOINT_COST   = .3;    // Other joints.

    for(auto i : root_dofs)
    {
        for(auto j : root_dofs)
        {
            eQ_joint(i,j) = 0;
        }
    }

    for(auto i : balance_dofs)
    {
        for(auto j : balance_dofs)
        {
            eQ_joint(i,j) *= BALANCE_COST;
        }
    }

    for(auto i : other_dofs)
    {
        for(auto j : other_dofs)
        {
            eQ_joint(i,j) *= OTHER_JOINT_COST;
        }
    }

    // Construct the Q matrix for position DoFs.
    eQpos = (BALANCE_COST * this->eQ_balance) + eQ_joint;
    #else
    for (int i = 0; i < 4; i++)
    {
        mj_jacSubtreeCom(m, d, jac_CoM, i); //eJac_CoM (3, model->nv)
        std::cout << "Jacobian " << i << std::endl;
        std::cout<< eJac_CoM << std::endl;
    }

    eQpos = Eigen::MatrixXd::Zero(m->nv, m->nv);
    #endif
    // No explicit penalty for velocities.
    eQ.topLeftCorner(m->nv, m->nv) = eQpos;
    eQ.topRightCorner(m->nv, m->nv) = Eigen::MatrixXd::Zero(m->nv, m->nv);
    eQ.bottomLeftCorner(m->nv, m->nv) = Eigen::MatrixXd::Zero(m->nv, m->nv);
    eQ.bottomRightCorner(m->nv, m->nv) = Eigen::MatrixXd::Zero(m->nv, m->nv);

    //==============================================================
    //          LQR -> A and B
    //==============================================================
    mjd_transitionFD(m, d, 1e-6, true, A, B, nullptr, nullptr);

    //==============================================================
    //          LQR -> P and K
    //==============================================================
    // Solve discrete Riccati equation.
    // P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    //https://github.com/TakaHoribe/Riccati_Solver/blob/master/riccati_solver.cpp

    this->eAT = eA.transpose(); //->  AX = influence of state
    this->eBT = eB.transpose(); //->  BU = influence of control
    this->eRinv = eR.inverse(); //->  actuator weights

    this->eP = this->eQ;

    uint16_t max_iterations = 10;
    for (uint16_t i = 0; i < max_iterations; ++i)
    {
        eAT_eP = eAT*eP;
        eBT_eP = eBT*eP;

        eK = ((eBT_eP*eB)+eR).inverse() * eBT_eP*eA;

        eP = eQ + eAT_eP*eA - eAT_eP*(eB*eK);
    }

    //==============================================================
    //         LQR -> K
    //==============================================================
    // K is the matrix used to correct the error in x
    // correctionForce = D @ error_in_state

    eBT_eP = eBT*eP;
    this->eK = ((eBT_eP*eB)+eR).inverse() * eBT_eP*eA;
    
    std::cout << "\neK\n" << eK << std::endl;
};

void Controller::loop(mjData* currentState)
{
    cpMjData(m, d, currentState);
    // //==============================================================
    // //                  Calculate position error
    // //==============================================================
    mj_differentiatePos(m, this->error_position, 1, targetPose, d->qpos);

    eError_state.leftCols(m->nv)    = eError_position;
    eError_state.rightCols(m->nv)   = eVelocityMatrix;

    // //==============================================================
    // //                  Apply control and step
    // //==============================================================
    eMotorForceMatrix = eTargetPoseControl - eK * eError_state.transpose();

    applyControll(currentState);
};

void Controller::applyControll(mjData* dst)
{
    for(int i = 0; i < m->nu; i++)
    {
        dst->ctrl[i] = d->ctrl[i];
    }
}
