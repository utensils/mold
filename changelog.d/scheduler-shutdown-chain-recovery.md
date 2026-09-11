- **Long Wan clips now survive a machine restart.** The multi-GPU scheduler
  keeps ownership of active clip leases while GPU workers stop, so an
  interrupted auto-chained video is parked for Resume instead of being marked
  failed and swept as temporary work.
