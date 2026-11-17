"""Full execution reuses the exact same public step implementations."""


def run_full(core, context, step1=None, step1_5=None, step2=None):
    step2 = dict(step2 or {})
    for name, options in (("step1", step1), ("step1_5", step1_5), ("step2", step2)):
        context.check_cancelled()
        child = core._execute(name, dict(options or {}))
        if child.success:
            context.complete(name, child)
            head_done = (not child.completed.get("head_camera", {}).get("skipped", False)
                         and dict(step1_5 or {}).get("prepare", True))
            if name == "step1_5" and head_done:
                # Step 1.5 finished at the same ready pose Step 2 starts from, and only the head
                # moved after that. Skip Step 2's joint ready pose so the arms stop travelling
                # back to it and straight out again.
                collection = dict(step2.get("collection", {}))
                collection["skip_joint_pose"] = True
                step2["collection"] = collection
        else:
            context.checkpoint(name, child)
            context.result.finish(child.status, child.error)
            return
