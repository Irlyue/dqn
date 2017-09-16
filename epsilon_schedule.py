class LinearSchedule(object):
    def __init__(self, schedule_steps, final_p, initial_p=1.0):
        self.schedule_steps = schedule_steps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(1.0, step * 1.0 / self.schedule_steps)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
