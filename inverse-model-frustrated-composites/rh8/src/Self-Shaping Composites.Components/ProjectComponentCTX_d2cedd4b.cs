using System;
using SD = System.Drawing;

using Rhino.Geometry;

using Grasshopper.Kernel;
using GH_IO.Serialization;

namespace RhinoCodePlatform.Rhino3D.Projects.Plugin.GH
{
  public sealed class ProjectComponentCTX_d2cedd4b : ProjectComponent_Base, IGH_ContextualComponent, IGH_DocumentOwner
  {
    static readonly string s_scriptDataId = "d2cedd4b-f6f5-4eec-bec6-2c1c0f5bbfc7";
    static readonly string s_scriptIconData = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAAAlwSFlzAAAOwgAADsIBFShKgAAAAYVJREFUSEtjYBgFo2DAga6urhK6GNVAjaX6tM0+ms9LbbTK0eUoAurq6lL9DuonH4Zp/v8Ypfn/WrDmv5nOGhdVVFRk0NWSDFJM1cI2eKh/eBulDTYchkH8jZ7qH3ItdUPQ9RAFtJWUrCc6qB47E6DxG9lgdHwmQPN3u51WP7p+nCDORDOm317t6qkAjd8vIrQwDMSGb4Vp/ZvppH4Y3SwUUGSj7bPYVf315SCN/x+iiDMYGT8I1/o/wUV3J7q5YODj4yOyLNbl63sSDX4dqfX/QZjm/9Nhhv8WBFm+8PPzk0I3Gw72r162aXuE5T90Q7DhyxEG/7bmhf9fX5nxZ2Vn48u+rp7a////M6GbiQHW5ScHbU71/v4gTOv/qwjN/x+iUVPO43DN/wcSHP9uX7/m99atW1PR9RMFQGm+0896/tKM0M/rS5P/rS9P+78jweXfrqzA/1s7qv8fPHz4w7179+TR9ZEFrly5YXPy9JlNp86ceX7i1Kn3F69c2Y+uZhSMAuoDADInIUB9hpUeAAAAAElFTkSuQmCC";

    public override Guid ComponentGuid { get; } = new Guid("d2cedd4b-f6f5-4eec-bec6-2c1c0f5bbfc7");

    public override GH_Exposure Exposure { get; } = GH_Exposure.primary;

    public override bool Obsolete { get; } = false;

    public ProjectComponentCTX_d2cedd4b() : base(GetResource(s_scriptDataId), s_scriptIconData,
        name: "Simulation.gh",
        nickname: "Simulate",
        description: @"",
        category: "Self-Shaping Composites",
        subCategory: "Default"
        )
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager _) { }
    protected override void RegisterOutputParams(GH_OutputParamManager _) { }

    protected override void BeforeSolveInstance() => m_script.BeforeSolve(this);

    protected override void SolveInstance(IGH_DataAccess DA) => m_script.Solve(this, DA);

    protected override void AfterSolveInstance() => m_script.AfterSolve(this);

    public override BoundingBox ClippingBox => m_script.GetClipBox(this);

    public override void DrawViewportWires(IGH_PreviewArgs args) => m_script.DrawWires(this, args);

    public override void DrawViewportMeshes(IGH_PreviewArgs args) => m_script.DrawMeshes(this, args);

    #region IGH_ContextualComponent
    GH_Archive IGH_ContextualComponent.Archive => m_script.Archive;
    #endregion

    #region IGH_DocumentOwner
    GH_Document IGH_DocumentOwner.OwnerDocument() => OnPingDocument();

    void IGH_DocumentOwner.DocumentClosed(GH_Document document)
    {
      // Internal docs are embedded, we don't need to do anything.
    }

    void IGH_DocumentOwner.DocumentModified(GH_Document document)
    {
      // Internal docs are embedded, we don't need to do anything.
    }
    #endregion
  }
}
