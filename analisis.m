  x = csvread ("./error.csv")
  u = csvread ("./actuation.csv")

  error                   = x(:, 1:1)
  actuation               = u(:, 1:1)
  #velocity           = x(:, 6:6) / 100
  
hold on
    %plot(time, velocity)
    plot(error)
    plot(actuation)
    %plot(time, diff)
    %legend({"velocity", "acceleration", "nextStepAcceleration", "diff"})
    legend({"error", "actuation"})
hold off